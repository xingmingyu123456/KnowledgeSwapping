import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

def vis_pis(image, targets,save_path,model,postprocessors,id2name,id2path):
    # image, targets = dataset_val[0]
    print(targets['labels'])
    box_label = [id2name[int(item)] for item in targets['labels']]
    gt_dict = {
        'boxes': targets['boxes'],
        'image_id': targets['image_id'],
        'size': targets['size'],
        'box_label': box_label,
    }

    vslzr = COCOVisualizer()
    # vslzr.visualize(image, gt_dict,id2path,savedir=save_path_before,show_in_console=False)

    output,_ = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    thershold = 0.3  # set a thershold

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold

    # test = [int(item) for item in labels[select_mask]]
    # print(test)
    # box_label = [id2name[int(item)] for item in labels[select_mask]]
    #coco的序号不连续
    new_select_mask =select_mask
    for index,item in enumerate(labels):
        if id2name.get(int(item)) is None:
            new_select_mask[index] =False

    box_label = [id2name[int(item)] for item in labels[new_select_mask]]


    pred_dict = {
        'boxes': boxes[new_select_mask],
        'size': targets['size'],
        'box_label': box_label,
        'image_id': targets['image_id'],
    }
    vslzr.visualize(image, pred_dict, id2path,savedir=save_path,show_in_console=False)



def visual(model_checkpoint_path,save_path,dataset_path):
    model_config_path = "config/DINO/DINO_4scale_swin_swap.py" # change the path of the model config file
    # model_checkpoint_path = "/data1/xmy/DINO/logs/swin4c01-22-17-11-17/checkpoint0009_forget.pth" # change the path of
    # save_path = "/data1/xmy/DINO/data/visual/coco0122/bench/dog_forget_after_vis"
    os.makedirs(save_path, exist_ok=True)

    args = SLConfig.fromfile(model_config_path)
    # args.coco_path = "/data1/xmy/DINO/data/coco0120/cocobench_forget" # the path of coco
    args.coco_path = dataset_path # the path of coco
    args.device = 'cuda'
    args.lora_rank=8
    args.lora_pos="FFN"
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    with open('util/coco_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k):v for k,v in id2name.items()}

    args.dataset_file = 'coco'

    args.fix_size = False

    dataset_val = build_dataset(image_set='val', args=args)
    root =dataset_val.root
    imgs = dataset_val.coco.imgs
    id2path = {}
    for id, img_dic in imgs.items():
        name = img_dic["file_name"]
        id2path[id]=os.path.join(root,name)
    for pic,targets in dataset_val:
        vis_pis(pic,targets,save_path,model,postprocessors,id2name,id2path)

def visual_start(model_checkpoint_path,save_path,dataset_path):
    model_config_path = "config/DINO/DINO_4scale_swin_swap.py" # change the path of the model config file
    # model_checkpoint_path = "/data1/xmy/DINO/logs/swin4c01-22-17-11-17/checkpoint0009_forget.pth" # change the path of
    # save_path = "/data1/xmy/DINO/data/visual/coco0122/bench/dog_forget_after_vis"
    os.makedirs(save_path, exist_ok=True)

    args = SLConfig.fromfile(model_config_path)
    # args.coco_path = "/data1/xmy/DINO/data/coco0120/cocobench_forget" # the path of coco
    args.coco_path = dataset_path # the path of coco
    args.device = 'cuda'
    args.lora_rank=0
    args.lora_pos="NO"
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    _ = model.eval()

    with open('util/coco_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k):v for k,v in id2name.items()}

    args.dataset_file = 'coco'

    args.fix_size = False

    dataset_val = build_dataset(image_set='val', args=args)
    root =dataset_val.root
    imgs = dataset_val.coco.imgs
    id2path = {}
    for id, img_dic in imgs.items():
        name = img_dic["file_name"]
        id2path[id]=os.path.join(root,name)
    for pic,targets in dataset_val:
        vis_pis(pic,targets,save_path,model,postprocessors,id2name,id2path)




if __name__ == '__main__':
    dataset_path = "/data1/xmy/DINO/data/coco0120/cocobench_remain"
    model_path_S = "/data1/xmy/DINO/logs/swin4c01-14-14-47-40-cl101/checkpoint0004.pth"
    save_path_S = "/data1/xmy/DINO/data/visual/coco0125/cub_remain_S"
    model_path_L = "/data1/xmy/DINO/logs/swin4c01-24-16-11-38-cub-flf/checkpoint0019_learn.pth"
    save_path_L = "/data1/xmy/DINO/data/visual/coco0125/cub_remain_FL"
    model_path_LF = "/data1/xmy/DINO/logs/swin4c01-24-16-11-38-cub-flf/checkpoint0019_forget.pth"
    save_path_LF =  "/data1/xmy/DINO/data/visual/coco0125/cub_remain_F"


    visual_start(model_path_S, save_path_S, dataset_path)
    visual(model_path_L, save_path_L, dataset_path)
    visual(model_path_LF, save_path_LF, dataset_path)
