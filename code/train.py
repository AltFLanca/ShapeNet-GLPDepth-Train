import os
import numpy as np
import cv2
from datetime import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
#from tensorboardX import SummaryWriter
from utils.writer import DataWriter

from models.model import GLPDepth
import utils.metrics as metrics
from utils.criterion import SiLogLoss
import utils.logging_ as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    # Logging
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = DataWriter(log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'artifacts')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = None

    # Load checkpoint if provided
    if args.ckpt_dir is not None:
        model = GLPDepth(max_depth=args.max_depth, is_train=False)
        model_weight = torch.load(args.ckpt_dir)
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        model.load_state_dict(model_weight)
        print('Loaded model from %s' % args.ckpt_dir)
    else:
        model = GLPDepth(max_depth=args.max_depth, is_train=True)

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset,
        'data_path': args.data_path,
        'train_txt_path': args.train_txt_path,
        'test_txt_path': args.test_txt_path,
    }

    if args.dataset != 'shapenetsem':
        dataset_kwargs['metadata_path'] = args.metadata_path

    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True)

    # Training settings
    criterion_d = SiLogLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    global global_step
    global_step = len(train_loader) * (args.start_epoch - 1)

    # Perform experiment
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_d, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args)
        writer.add_scalar('Training loss', loss_train)
        
        results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                            device=device, epoch=epoch, args=args,
                                            log_dir=log_dir)
        writer.add_scalar('Val loss', loss_val)

        result_lines = logging.display_result(results_dict)
        print(result_lines)

        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
            txtfile.write(result_lines)                

        for each_metric, each_results in results_dict.items():
            writer.add_scalar(each_metric, each_results)
    
    writer.save()


def train(train_loader, model, criterion_d, optimizer, device, epoch, args):
    torch.cuda.empty_cache()  
    global global_step
    model.train()
    depth_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    train_loader_len = len(train_loader)

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
    max_lr = 1e-4
    mid_lr = 6.5e-5
    min_lr = 3e-5

    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        for param_group in optimizer.param_groups:
            if global_step < train_loader_len * half_epoch:
                current_lr = (mid_lr - min_lr) * (global_step /
                                              train_loader_len/half_epoch) ** 0.9 + min_lr
            else:
                current_lr = (min_lr - max_lr) * (global_step /
                                              train_loader_len/half_epoch - 1) ** 0.9 + max_lr
            param_group['lr'] = current_lr

        input_RGB = batch['image'].to(device)
        depth_gt = batch.get('depth_normalized', batch['depth']).to(device)

        preds = model(input_RGB)

        optimizer.zero_grad()
        loss_d = criterion_d(preds['pred_d'].squeeze(1), depth_gt)
        
        # Check if loss_d is NaN
        if not torch.isnan(loss_d):
            depth_loss.update(loss_d.item(), input_RGB.size(0))
            loss_d.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                            ('Depth Loss: %.4f (%.4f)' %
                            (depth_loss.val, depth_loss.avg)))
            
            optimizer.step()
        else:
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                            ('Depth Loss: %s (%.4f)' %
                            ('NaN', depth_loss.avg)))

    return depth_loss.avg


@torch.no_grad()
def validate(val_loader, model, criterion_d, device, epoch, args, log_dir):
    depth_loss = logging.AverageMeter()
    model.eval()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            log_dir, 'epoch_%02d_model.ckpt' % epoch))

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        filename = batch['filename'][0]

        with torch.no_grad():
            preds = model(input_RGB)
        
        pred_d = preds['pred_d'].squeeze(1)
        if 'normalization' in batch:
            normalization = batch['normalization'].to(device=device, dtype=pred_d.dtype).view(-1, 1, 1)
            pred_d = pred_d * normalization
        depth_gt = batch['depth'].to(device)
        loss_d = criterion_d(pred_d, depth_gt)
        pred_d, depth_gt = pred_d.squeeze(0), depth_gt.squeeze(0)

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        save_path = os.path.join(result_dir, filename)

        if save_path.split('.')[-1] == 'jpg':
            save_path = save_path.replace('jpg', 'png')

        if args.save_result:
            pred_d_numpy = pred_d.squeeze().detach().cpu().numpy() * 1000.0
            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loss_d = depth_loss.avg
        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics, loss_d


if __name__ == '__main__':
    main()
