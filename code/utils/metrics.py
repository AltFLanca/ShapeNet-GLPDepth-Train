import torch


def eval_depth(pred, target):
    assert pred.shape == target.shape
    
    if pred.shape == torch.Size([0]):
        return {'d1': 1.0, 'd2': 1.0, 'd3': 1.0, 'abs_rel': 0.0,
                'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0,
                'log10': 0.0, 'silog': 0.0}

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': 1.0 if torch.isnan(d1).item() else d1.item(),
            'd2': 1.0 if torch.isnan(d2).item() else d2.item(),
            'd3': 1.0 if torch.isnan(d3).item() else d3.item(),
            'abs_rel': 0.0 if torch.isnan(abs_rel).item() else abs_rel.item(),
            'sq_rel': 0.0 if torch.isnan(sq_rel).item() else sq_rel.item(),
            'rmse': 0.0 if torch.isnan(rmse).item() else rmse.item(),
            'rmse_log': 0.0 if torch.isnan(rmse_log).item() else rmse_log.item(),
            'log10': 0.0 if torch.isnan(log10).item() else log10.item(),
            'silog': 0.0 if torch.isnan(silog).item() else silog.item()}


def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred = pred.detach().clone()
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
    eval_mask[45:471, 41:601] = 1
    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]