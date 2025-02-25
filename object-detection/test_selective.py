# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import loralib as lora
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
from datetime import timedelta

from iter_engine import train_one_epoch_learn_iter
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
import torch.distributed as dist
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test,train_one_epoch_learn,train_one_epoch_forget
from datasets.coco import build_learn,build_forget,build_remain
from util.utils import reinitialize_lora_parameters
from torch.utils.data import ConcatDataset
from engine import train_one_epoch_learn_test
def mergedataset(dataset_train_learn,dataset_train_remain):
    pass
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('-config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--ococo_panptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # lora微调
    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--lora_pos', type=str, default='FFN')
    # 学习集合 遗忘集 路径
    parser.add_argument("--remain_path", type=str, default="")
    parser.add_argument("--forget_path", type=str, default="")
    parser.add_argument("--learn_path", type=str, default="")

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 回复断点文件夹
    parser.add_argument('--resume_dir', default="")

    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser



def get_structure_loss(model: torch.nn.Module):
    if isinstance(model, (torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
        model_without_ddp = model.module

    else:
        model_without_ddp = model
    # if isinstance(model, torch.nn.DataParallel):
    #     model_without_ddp = model.module
    # else:
    #     model_without_ddp = model
    learnable_params_name = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]
    # print("learnable_params_name",learnable_params_name)
    # learnable_params_name = [
    #     name
    #     for name, param in model_without_ddp.named_parameters()
    # ]
    # print("learnable_params_name",learnable_params_name)
    group_layers = []

    for i in range(6):
        group_item = []
        group_item.append("transformer.encoder.layers.{}.linear1.lora_A".format(i))
        group_item.append("transformer.encoder.layers.{}.linear1.lora_B".format(i))
        group_item.append("transformer.encoder.layers.{}.linear2.lora_A".format(i))
        group_item.append("transformer.encoder.layers.{}.linear2.lora_B".format(i))
        group_layers.append(group_item)
    for i in range(6):
        group_item = []
        group_item.append("transformer.decoder.layers.{}.linear1.lora_A".format(i))
        group_item.append("transformer.decoder.layers.{}.linear1.lora_B".format(i))
        group_item.append("transformer.decoder.layers.{}.linear2.lora_A".format(i))
        group_item.append("transformer.decoder.layers.{}.linear2.lora_B".format(i))
        group_layers.append(group_item)
    layers_num=[2,2,18,2]
    for i in range(4):
        for j in range(layers_num[i]):
            group_item = []
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc1.lora_A")
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc1.lora_B")
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc2.lora_A")
            group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc2.lora_B")
            group_layers.append(group_item)

    # get the parameters
    params_dict = dict(model_without_ddp.named_parameters())

    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(
                params_dict.get(item)
                if item in learnable_params_name
                else None
            )
        group_params.append(group_param)

    def group_sparse_multi_module(group_param):
        # group_param is a list of parameters
        # calculate the loss for a single group of parameters
        def l2_loss(param_group):
            return torch.sum(param_group**2)

        lasso_sum = 0
        for param in group_param:
            lasso_sum += l2_loss(param)
        return torch.sqrt(lasso_sum)

    group_sparse_loss = 0
    # calculate the loss for all groups of parameters
    for group_param in group_params:
        group_sparse_loss += group_sparse_multi_module(group_param)
    if torch.distributed.is_initialized():
        dist.all_reduce(group_sparse_loss, op=dist.ReduceOp.SUM)
        group_sparse_loss /= torch.distributed.get_world_size()
    # print('group_sparse_loss', group_sparse_loss)
    return group_sparse_loss



def get_norm_of_lora(
    model, type="L2", group_num=6, group_type: str = "block", group_pos: str = "FFN"
):
    """
    get L2 norm of each group of lora parameters
    :param model: model (is already without ddp)
    :param type: L2 or L1
    :param group_num: 6 or 12 or 18
    :param group_type:
        -block (each Transformer block is a group)
        -lora (each LoRA is a group), 2 LoRAs in one block
        -matrix (each layer is a group), 2 matrix in one LoRA
    :param group_pos: lora pos
    :return: norm_list, list of norm of each group, type: list of tensor.float with length 12
    """
    if isinstance(model, (torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
        model_without_ddp = model.module

    else:
        model_without_ddp = model
    model = model_without_ddp
    with torch.no_grad():
        norm_list = []
        group_layers = []

        if group_pos == "FFN":
            if group_type == "block":
                layers_num = [2, 2, 18, 2]
                for i in range(4):
                    for j in range(layers_num[i]):
                        group_item = []
                        group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc1.lora_A")
                        group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc1.lora_B")
                        group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc2.lora_A")
                        group_item.append(f"backbone.0.layers.{i}.blocks.{j}.mlp.fc2.lora_B")
                        group_layers.append(group_item)
                for i in range(6):
                    group_item = []
                    group_item.append("transformer.encoder.layers.{}.linear1.lora_A".format(i))
                    group_item.append("transformer.encoder.layers.{}.linear1.lora_B".format(i))
                    group_item.append("transformer.encoder.layers.{}.linear2.lora_A".format(i))
                    group_item.append("transformer.encoder.layers.{}.linear2.lora_B".format(i))
                    group_layers.append(group_item)
                for i in range(6):
                    group_item = []
                    group_item.append("transformer.decoder.layers.{}.linear1.lora_A".format(i))
                    group_item.append("transformer.decoder.layers.{}.linear1.lora_B".format(i))
                    group_item.append("transformer.decoder.layers.{}.linear2.lora_A".format(i))
                    group_item.append("transformer.decoder.layers.{}.linear2.lora_B".format(i))
                    group_layers.append(group_item)

                # for i in range(group_num):
                #     group_item = []
                #     group_item.append(
                #         "transformer.encoder.layer.{}.ffn.fc1.lora_A".format(i)
                #     )
                #     group_item.append(
                #         "transformer.encoder.layer.{}.ffn.fc1.lora_B".format(i)
                #     )
                #     group_item.append(
                #         "transformer.encoder.layer.{}.ffn.fc2.lora_A".format(i)
                #     )
                #     group_item.append(
                #         "transformer.encoder.layer.{}.ffn.fc2.lora_B".format(i)
                #     )
                #     group_layers.append(group_item)
            elif group_type == "lora":
                for i in range(group_num):
                    group_item = []
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.0.lora_A".format(i)
                    )
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.0.lora_B".format(i)
                    )
                    group_layers.append(group_item)
                for i in range(group_num):
                    group_item = []
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.3.lora_A".format(i)
                    )
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.3.lora_B".format(i)
                    )
                    group_layers.append(group_item)
            elif group_type == "matrix":
                for i in range(group_num):
                    group_item = []
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.0.lora_A".format(i)
                    )
                    group_layers.append(group_item)
                for i in range(group_num):
                    group_item = []
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.0.lora_B".format(i)
                    )
                    group_layers.append(group_item)
                for i in range(group_num):
                    group_item = []
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.3.lora_A".format(i)
                    )
                    group_layers.append(group_item)
                for i in range(group_num):
                    group_item = []
                    group_item.append(
                        "transformer.layers.{}.1.fn.fn.net.3.lora_B".format(i)
                    )
                    group_layers.append(group_item)

        elif group_pos == "Attention":
            for i in range(group_num):
                group_item = []
                group_item.append(
                    "transformer.layers.{}.0.fn.fn.to_qkv.lora_A".format(i)
                )
                group_item.append(
                    "transformer.layers.{}.0.fn.fn.to_qkv.lora_B".format(i)
                )
                group_layers.append(group_item)

        print("\033[31mgroup_layers_names\033[0m\n", group_layers)
        # get the parameters
        group_params = []
        for group_item in group_layers:
            group_param = []
            for item in group_item:
                group_param.append(model.state_dict()[item])
            group_params.append(group_param)
        # print('group_parmas ', group_params)

        for group_param in group_params:
            if type == "L2":
                norm = 0
                length = len(group_param)
                for i in range(length):
                    norm += torch.norm(group_param[i])
                norm_list.append(norm)
            elif type == "L1":
                norm = 0
                length = len(group_param)
                for i in range(length):
                    norm += torch.norm(group_param[i], p=1)
                norm_list.append(norm)
            else:
                raise ValueError("type should be L1 or L2")
        num_list = []
        for item in norm_list:
            num_list.append(item.item())
        return num_list


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def record(logger,stats,epoch=None):
    stats_all = stats['coco_eval_bbox']
    metric_names = [
        'AP @ IoU=0.50:0.95 | area=all | maxDets=100',
        'AP @ IoU=0.50 | area=all | maxDets=100',
        'AP @ IoU=0.75 | area=all | maxDets=100',
        'AP @ IoU=0.50:0.95 | area=small | maxDets=100',
        'AP @ IoU=0.50:0.95 | area=medium | maxDets=100',
        'AP @ IoU=0.50:0.95 | area=large | maxDets=100',
        'AR @ IoU=0.50:0.95 | area=all | maxDets=1',
        'AR @ IoU=0.50:0.95 | area=all | maxDets=10',
        'AR @ IoU=0.50:0.95 | area=all | maxDets=100',
        'AR @ IoU=0.50:0.95 | area=small | maxDets=100',
        'AR @ IoU=0.50:0.95 | area=medium | maxDets=100',
        'AR @ IoU=0.50:0.95 | area=large | maxDets=100'
    ]
    logger.info(f"{'=' * 20} COCO Evaluation Metrics {'=' * 20}")
    if epoch is not None:
        logger.info(f"Epoch: {epoch}")
    for name, value in zip(metric_names, stats_all):
        logger.info(f"{name}: {value:.3f}")
    stats_single = stats['category_ap']
    stats_filter = {name:ap for name,ap in stats_single.items() if ap != -1}
    print(stats_filter)
    logger.info(stats_filter)

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # build model
    model, criterion, postprocessors = build_model_main(args)
    device = torch.device("cpu")
    model.to(device)


    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
        #                                                   find_unused_parameters=args.find_unused_params)
        # model = torch.nn.parallel.DataParallel(model, device_ids=[args.gpu])
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params,
            broadcast_buffers=False
        )
        model_without_ddp = model.module

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        _tmp_st = OrderedDict(
            {k: v for k, v   in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        # logger.info(str(_load_output))

    if args.lora_rank > 0:
        lora.mark_only_lora_as_trainable(model_without_ddp)
        print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)

    else:
        print(
            "Do not use LoRA in Transformer FFN, train all parameters."
        )

    norm_list = get_norm_of_lora(model_without_ddp, type="L2", group_num=36, group_type="block", group_pos="FFN")
    # print(norm_list)
    return norm_list





if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # new_args = copy.deepcopy(args)
    # save_path = "/data1/xmy/DINO/data/compare/cub-fftl.json"
    # learn_path = "/data1/xmy/DINO/logs/swin4c01-24-16-11-38-cub-flf/checkpoint0019_learn.pth"
    # forget_path = "/data1/xmy/DINO/logs/swin4c01-24-16-11-38-cub-flf/checkpoint0019_forget.pth"
    # args.pretrain_model_path = learn_path
    # norm_list_learn=main(args)
    #
    #
    # new_args.pretrain_model_path = forget_path
    # norm_list_forget=main(new_args)
    #
    # print("norm_list_learn", norm_list_learn)
    # print("norm_list_forget",norm_list_forget)
    #
    #
    #
    # data = {
    #     "norm_list_learn":norm_list_learn,
    #     "norm_list_forget": norm_list_forget,
    # }
    #
    # with open(save_path, "w") as f:
    #     json.dump(data, f)

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    new_args = copy.deepcopy(args)
    save_path = "/data1/xmy/DINO/data/compare/cub-flf_forget.json"
    forget_path = "/data1/xmy/DINO/logs/swin4c01-24-16-11-38-cub-flf/checkpoint0019_forget_twice.pth"

    new_args.pretrain_model_path = forget_path
    norm_list_forget=main(new_args)
    print("norm_list_forget",norm_list_forget)

    data = {
        "norm_list_forget": norm_list_forget,
    }

    with open(save_path, "w") as f:
        json.dump(data, f)