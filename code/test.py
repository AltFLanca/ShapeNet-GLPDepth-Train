import os
import numpy as np
import cv2
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging_ as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.exp_name)
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)
    
    if args.do_evaluate:
        result_metrics = {}
        for metric in metric_name:
            result_metrics[metric] = 0.0

    print("\n1. Define Model")
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n2. Define Dataloader")
    dataset_kwargs = {
        'dataset_name': args.dataset,
        'data_path': args.data_path,
        'train_txt_path': args.train_txt_path,
        'test_txt_path': args.test_txt_path,
        'is_train': False,
    }

    if args.dataset != 'shapenetsem':
        dataset_kwargs['metadata_path'] = args.metadata_path

    test_dataset = get_dataset(**dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)

    print("\n3. Inference & Evaluate")
    for batch_idx, batch in enumerate(test_loader):
        input_RGB = batch['image'].to(device)
        filename = batch['filename']

        with torch.no_grad():
            pred = model(input_RGB)
        pred_d = pred['pred_d'].squeeze(1)
        if 'normalization' in batch:
            normalization = batch['normalization'].to(device=device, dtype=pred_d.dtype).view(-1, 1, 1)
            pred_d = pred_d * normalization
        pred_d = pred_d.squeeze(0)

        if args.do_evaluate:
            depth_gt = batch['depth'].to(device).squeeze(0)
            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)

            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]
                    

        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename[0])
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            png_depth = pred_d.cpu().numpy() * 1000.0
            cv2.imwrite(save_path, png_depth.astype(np.uint16),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        if args.save_visualize:
            save_path = os.path.join(result_path, filename[0])
            pred_d_numpy = pred_d.squeeze().cpu().numpy()
            pred_d_numpy = (pred_d_numpy / max(float(pred_d_numpy.max()), 1e-8)) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)
            pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
            cv2.imwrite(save_path, pred_d_color)
        logging.progress_bar(batch_idx, len(test_loader), 1, 1)

    if args.do_evaluate:
        for key in result_metrics.keys():
            result_metrics[key] = result_metrics[key] / (batch_idx + 1)
        display_result = logging.display_result(result_metrics)
        print(display_result)

    print("Done")


if __name__ == "__main__":
    main()
