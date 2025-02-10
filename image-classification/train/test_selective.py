import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import sys
import random
import numpy as np
import numpy as np
np.bool = bool  # 临时兼容
import os
import ml_collections
from datetime import datetime
from torch.utils.data import Dataset
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
sys.path.append(os.path.join(current_dir, '..'))
from image_iter import CLDatasetWrapper, CustomSubset,MergedDataset
from config import get_config
from util.utils import (
    AverageMeter,
)
import torch
import  numpy
from util.utils import (
    split_dataset,
    count_trainable_parameters,
    reinitialize_lora_parameters,
)
from util.args import get_args
from vit_pytorch_face import ViT_bird
# from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import loralib as lora
from engine_cl import train_one_epoch, eval_data, train_one_epoch_regularzation,train_one_epoch_learn
from torch.utils.data import Subset
from util.cal_norm import get_norm_of_lora
from torch.utils.data import DataLoader
from customlogger import CustomLogger
import math
import json


def get_structure_loss(model: torch.nn.Module):
    if isinstance(model, torch.nn.DataParallel):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    learnable_params_name = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]
    # learnable_params_name = [
    #     name
    #     for name, param in model_without_ddp.named_parameters()
    # ]
    # print("learnable_params_name",learnable_params_name)
    group_layers = []
    """
    transformer.layers.0.1.fn.fn.net.0.lora_A
    transformer.layers.0.1.fn.fn.net.0.lora_B
    transformer.layers.0.1.fn.fn.net.3.lora_A
    transformer.layers.0.1.fn.fn.net.3.lora_B
    transformer.layers.1.1.fn.fn.net.0.lora_A
    transformer.layers.1.1.fn.fn.net.0.lora_B
    transformer.layers.1.1.fn.fn.net.3.lora_A
    transformer.layers.1.1.fn.fn.net.3.lora_B
    transformer.layers.2.1.fn.fn.net.0.lora_A
    transformer.layers.2.1.fn.fn.net.0.lora_B
    transformer.layers.2.1.fn.fn.net.3.lora_A
    transformer.layers.2.1.fn.fn.net.3.lora_B
    transformer.layers.3.1.fn.fn.net.0.lora_A
    transformer.layers.3.1.fn.fn.net.0.lora_B
    transformer.layers.3.1.fn.fn.net.3.lora_A
    transformer.layers.3.1.fn.fn.net.3.lora_B
    transformer.layers.4.1.fn.fn.net.0.lora_A
    transformer.layers.4.1.fn.fn.net.0.lora_B
    transformer.layers.4.1.fn.fn.net.3.lora_A
    transformer.layers.4.1.fn.fn.net.3.lora_B
    transformer.layers.5.1.fn.fn.net.0.lora_A
    transformer.layers.5.1.fn.fn.net.0.lora_B
    transformer.layers.5.1.fn.fn.net.3.lora_A
    transformer.layers.5.1.fn.fn.net.3.lora_B
    """
    for i in range(12):
        group_item = []
        group_item.append("transformer.encoder.layer.{}.ffn.fc1.lora_A".format(i))
        group_item.append("transformer.encoder.layer.{}.ffn.fc1.lora_B".format(i))
        group_item.append("transformer.encoder.layer.{}.ffn.fc2.lora_A".format(i))
        group_item.append("transformer.encoder.layer.{}.ffn.fc2.lora_B".format(i))
        group_layers.append(group_item)

    # get the parameters
    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(
                model_without_ddp.get_parameter(item)
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
                for i in range(12):
                    group_item = []
                    group_item.append("transformer.encoder.layer.{}.ffn.fc1.lora_A".format(i))
                    group_item.append("transformer.encoder.layer.{}.ffn.fc1.lora_B".format(i))
                    group_item.append("transformer.encoder.layer.{}.ffn.fc2.lora_A".format(i))
                    group_item.append("transformer.encoder.layer.{}.ffn.fc2.lora_B".format(i))
                    group_layers.append(group_item)
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


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_norm_list_from_check(model,path):
    checkpoint = torch.load(path)
    if isinstance(model_learn, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    norm_list = get_norm_of_lora(model, type="L2", group_num=12,group_type="block",group_pos="FFN")
    return norm_list


if __name__ == "__main__":


    config = get_b16_config()
    model_learn = ViT_bird(config, 224, zero_head=True, num_classes=200)
    model_forget = ViT_bird(config, 224, zero_head=True, num_classes=200)

    learn_path = "/home/xmy/code/classify-bird/compare/fltf_crop_disease_10e_re/learn_10.pth"
    forget_path = "/home/xmy/code/classify-bird/compare/fftl_crop_disease_10e_re/forget_10.pth"
    json_save_path = "/home/xmy/code/classify-bird/compare/two_re/compare_crop_disease.json"
    norm_list_learn = get_norm_list_from_check(model_learn,learn_path)
    norm_list_forget = get_norm_list_from_check(model_forget,forget_path)

    data = {
        "norm_list_learn": norm_list_learn,
        "norm_list_forget": norm_list_forget
    }
    print(data)
    with open(json_save_path, "w") as f:
        json.dump(data, f)

