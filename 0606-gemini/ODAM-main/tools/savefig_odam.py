import os
import sys
import math
import argparse
import json
import cv2

import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision.ops import nms
import torch.nn as nn
from typing import List

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')
from data.Coco import COCODataset
from utils import misc_utils, pytorch_nms_utils
from det_oprs import bbox_opr


eps = torch.finfo(torch.float32).eps

def eval_all(args, config, network):
    # model_path
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    model_file = os.path.join(saveDir,
            'dump-{}.pth'.format(args.resume_weights))
    print(model_file)
    # The new DETR network handles weight loading internally
    # assert os.path.exists(model_file)

    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)
	# load data
    img_folder = config.eval_folder
    source = config.eval_source
    coco = COCODataset(config, img_folder, source, is_train=False)
    print(coco.__len__())
    len_dataset = coco.__len__()
    config.drise_output_format = False
    figdir = './visualization/odam/'
    inference(config, network, model_file, coco, 0, len_dataset, figdir, args)


def ml_nms(boxlist, nms_th, max_num):
    boxes = boxlist[:,:4]
    scores = boxlist[:,4]
    labels = boxlist[:,5]
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(labels).cpu().tolist()):
        mask = (labels == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], nms_th)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    if len(keep) > max_num:
        keep = keep[:max_num]
    return keep

def inference(config, network, model_file, dataset, start, end, figdir, args):
    torch.set_default_tensor_type('torch.FloatTensor')

    # Check if this is DETR model by checking the function name passed
    is_detr = hasattr(network, '__name__') and network.__name__ == 'get_network'

    if is_detr:
        net = network(args) # DETR uses get_network factory function
        net.cuda()
        net = net.eval()
        # Weight loading is handled inside get_network for DETR
    else:
        # For traditional models
        net = network(config)
        net.cuda()
        check_point = torch.load(model_file)
        net.load_state_dict(check_point['state_dict'], strict=False)
        net = net.eval()

    # init data
    dataset.ids = dataset.ids[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)

	# inference
    coco_id2id = config.coco_id2id
    class_names = config.class_names

    pbar = tqdm(total=end-start, ncols=50)
    for (image, im_info, ID, anno, image_ori) in data_iter:
        image_ori = image_ori[0].numpy()
        image_ori = cv2.cvtColor(image_ori,cv2.COLOR_RGB2BGR)

        model_outputs = net(image.cuda(), im_info.cuda())

        # Handle different output formats
        if is_detr:
            # DETR network returns a single tensor with all info
            if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                 pred_boxes, features, levels = model_outputs
            else:
                 # Fallback for simplified return
                 pred_boxes = model_outputs
                 features, levels = {}, torch.zeros(len(pred_boxes)).cuda()

            if len(pred_boxes) > 0:
                basic_boxes = pred_boxes[:, :6]  # x1, y1, x2, y2, score, label
                cam_h = int(pred_boxes[0, -2].item())
                cam_w = int(pred_boxes[0, -1].item())
                cam_features = pred_boxes[:, 6:-2]
            else:
                basic_boxes = torch.empty(0, 6).cuda()
                cam_features = None
        else:
            # Traditional FCOS/RCNN format
            pred_boxes, features, levels = model_outputs
            basic_boxes = pred_boxes
            cam_features = None

        # NMS (not typically needed for DETR, but kept for consistency)
        if not is_detr:
            keep = ml_nms(basic_boxes, 0.5, 100)
            basic_boxes = basic_boxes[keep]
            levels = levels[keep]
            if features:
                 features['fpn'] = [f[keep] for f in features['fpn']]


        # select high confidence ones
        keep = basic_boxes[:, 4] > 0.1
        basic_boxes = basic_boxes[keep]
        if levels.numel() > 0: levels = levels[keep]
        if cam_features is not None: cam_features = cam_features[keep]

        if len(basic_boxes) == 0:
            pbar.update(1)
            continue

        pred_boxes_detach = basic_boxes.detach().cpu()
        pred_classes = pred_boxes_detach[:, 5]
        pred_scores = basic_boxes[:, 4]

        ori_imgsize = im_info[0, -2:]
        h,w = int(ori_imgsize[0]), int(ori_imgsize[1])
        resize = transforms.Resize([h,w])

        # get gts
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = [obj["bbox"] for obj in anno]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
        if len(gt_boxes) > 0:
            gt_boxes[:,2:4] += gt_boxes[:,0:2]
        gt_classes = [coco_id2id[int(obj["category_id"])] for obj in anno]
        gt_classes = torch.tensor(gt_classes)

        if len(gt_boxes) == 0 or len(pred_boxes_detach) == 0:
            pbar.update(1)
            continue

        overlaps = bbox_opr.box_overlap_opr(pred_boxes_detach[:,:4], gt_boxes)
        class_mask = pred_classes.unsqueeze(1) == gt_classes.unsqueeze(0)
        overlaps = overlaps*class_mask

        gt_inds, box_inds = [], []
        for j in range(min(len(pred_boxes_detach), len(gt_boxes))):
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            if gt_ovr <= 0.9: break

            box_ind = argmax_overlaps[gt_ind]
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
            gt_inds.append(gt_ind)
            box_inds.append(box_ind)

        box_inds = torch.tensor(box_inds).long()
        gt_inds = torch.tensor(gt_inds).long()

        if len(box_inds)>0:
            for i, box_ind in enumerate(box_inds):
                gt_ind = gt_inds[i]
                odam_map = None

                if is_detr:
                    if cam_features is not None:
                        odam_map = cam_features[box_ind].view(cam_h, cam_w)
                        odam_map = resize(odam_map.unsqueeze(0)).squeeze(0)
                        odam_map = (odam_map - odam_map.min()) / (odam_map.max() - odam_map.min()).clamp(min=eps)
                        odam_map = odam_map.cpu()
                else: # FCOS/RCNN logic
                    fm = features['fpn'][levels[box_ind]]
                    out_score = pred_scores[box_ind]
                    with torch.enable_grad():
                        fm.requires_grad = True
                        grad = torch.autograd.grad(out_score, fm, retain_graph=True)[0]
                    odam_map = F.relu_((grad * fm.detach()).sum(1))
                    odam_map = resize(odam_map).squeeze(0)
                    odam_map = (odam_map - odam_map.min()) / (odam_map.max() - odam_map.min()).clamp(min=eps)
                    odam_map = odam_map.cpu()

                if odam_map is not None:
                    if not os.path.exists(figdir): os.makedirs(figdir)
                    np.save(figdir+'{}_{}_{}.npy'.format(int(ID), int(gt_ind), 'npy'), odam_map)

                    gt_box = gt_boxes[gt_ind]
                    tmp_img = cv2.rectangle(image_ori.copy(), (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255,0,0), 2)
                    cv2.imwrite(figdir+'{}_{}_{}.jpg'.format(int(ID), int(gt_ind), 'object'), tmp_img)

                    pred_box = pred_boxes_detach[box_ind]
                    color = cv2.applyColorMap((odam_map.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                    c_ret = np.clip(image_ori * 0.5 + color * 0.5, 0, 255).astype(np.uint8)
                    c_ret = cv2.rectangle(c_ret, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0,0,255), 2)
                    cv2.imwrite(figdir+'{}_{}_{}.jpg'.format(int(ID), int(gt_ind), 'odamscore'), c_ret)

        pbar.update(1)

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()
    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config_coco import config

    # Handle different model types
    if args.model_dir == 'odam_detr':
        from network import get_network
        eval_all(args, config, get_network)
    else:
        from network import Network
        eval_all(args, config, Network)

if __name__ == '__main__':
    run_test()