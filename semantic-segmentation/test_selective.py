import json

import torch
from detectron2.engine import DefaultTrainer
import warnings

from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from torch.optim.lr_scheduler import StepLR

from testmapper import testMapper

# 忽略所有警告ss
warnings.filterwarnings("ignore")
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader,build_detection_test_loader

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# MaskDINO
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    CustomMaskFormerSemanticDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)

def get_structure_loss(model: torch.nn.Module):
    # DistributedDataParallel 是这个类型
    if isinstance(model, (torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
        model_without_ddp = model.module

    else:
        model_without_ddp = model
    learnable_params_name = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]
    # print("learable_params_name",learnable_params_name)
    # learnable_params_name = [
    #     name
    #     for name, param in model_without_ddp.named_parameters()
    # ]
    # print("learnable_params_name",learnable_params_name)
    group_layers = []

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
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.lora_A")
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.lora_B")
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.lora_A")
        group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.lora_B")
        group_layers.append(group_item)

    for i in range(9):
        group_item = []
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.lora_A")
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.lora_B")
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.lora_A")
        group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.lora_B")
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
                layers_num = [2, 2, 18, 2]
                for i in range(4):
                    for j in range(layers_num[i]):
                        group_item = []
                        group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc1.lora_A")
                        group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc1.lora_B")
                        group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc2.lora_A")
                        group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc2.lora_B")
                        group_layers.append(group_item)
                # layers = [0, 1, 3]
                # for i in layers:
                #     for j in range(2):
                #         group_item = []
                #         group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc1.lora_A")
                #         group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc1.lora_B")
                #         group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc2.lora_A")
                #         group_item.append(f"backbone.layers.{i}.blocks.{j}.mlp.fc2.lora_B")
                #         group_layers.append(group_item)
                #
                # for j in range(18):  # 原来这里可能写错了用的是2（用的是cityscape的时候）
                #     group_item = []
                #     group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc1.lora_A")
                #     group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc1.lora_B")
                #     group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc2.lora_A")
                #     group_item.append(f"backbone.layers.2.blocks.{j}.mlp.fc2.lora_B")
                #     group_layers.append(group_item)

                for i in range(6):
                    group_item = []
                    group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.lora_A")
                    group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.lora_B")
                    group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.lora_A")
                    group_item.append(f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.lora_B")
                    group_layers.append(group_item)

                for i in range(9):
                    group_item = []
                    group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.lora_A")
                    group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.lora_B")
                    group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.lora_A")
                    group_item.append(f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.lora_B")
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

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg



if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_swap.yaml"
    cfg = setup(args)
    save_path = "/home/xmy/code/Mask2Former-main/exp/compare_new/fltf-voc.json"
    checkpoint_forget = torch.load("/data1/xmy/Mask2Former-main/exp/log/2024-12-28-19-22-44-voc-forget/forget_model_0007000.pth")
    checkpoint_learn = torch.load("/data1/xmy/Mask2Former-main/exp/log/2024-12-28-16-51-37-voc-learn/learn_model_0007000.pth")


    model_forget = DefaultTrainer.build_model(cfg)
    model_learn = DefaultTrainer.build_model(cfg)
    if isinstance(model_forget, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        missing_keys, unexpected_keys = model_forget.module.load_state_dict(checkpoint_forget["model"], strict=False)
    else:
        missing_keys, unexpected_keys = model_forget.load_state_dict(checkpoint_forget["model"], strict=False)

    norm_list_forget = get_norm_of_lora(model_forget, type="L2", group_num=35,group_type="block",group_pos="FFN")

    if isinstance(model_learn, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        missing_keys, unexpected_keys = model_learn.module.load_state_dict(checkpoint_learn["model"], strict=False)
    else:
        missing_keys, unexpected_keys = model_learn.load_state_dict(checkpoint_learn["model"], strict=False)

    norm_list_learn = get_norm_of_lora(model_learn, type="L2", group_num=35, group_type="block", group_pos="FFN")

    data = {
        "norm_list_forget": norm_list_forget,
        "norm_list_learn": norm_list_learn
    }
    # data = {
    #     "norm_list_forget": norm_list_forget,
    # }

    with open(save_path, "w") as f:
        json.dump(data, f)




    # save_path = "/home/xmy/code/Mask2Former-main/exp/compare/flf-pet.json"
    # checkpoint1 = torch.load("/home/xmy/code/Mask2Former-main/exp/log/2025-01-15-03-06-25/forget_model_0007000.pth")
    #
    #
    # model1 = DefaultTrainer.build_model(cfg)
    # if isinstance(model1, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
    #     missing_keys, unexpected_keys = model1.module.load_state_dict(checkpoint1["model"], strict=False)
    # else:
    #     missing_keys, unexpected_keys = model1.load_state_dict(checkpoint1["model"], strict=False)
    #
    # norm_list_forget = get_norm_of_lora(model1, type="L2", group_num=35,group_type="block",group_pos="FFN")
    #
    #
    # data = {
    #     "norm_list_forget": norm_list_forget,
    # }
    #
    # with open(save_path, "w") as f:
    #     json.dump(data, f)





