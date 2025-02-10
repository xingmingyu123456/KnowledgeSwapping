# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticDatasetMapper","CustomMaskFormerSemanticDatasetMapper"]


class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict


class CustomMaskFormerSemanticDatasetMapper(MaskFormerSemanticDatasetMapper):

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]

        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )

        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))

        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        return cls(
            is_train=is_train,
            augmentations=augs,
            image_format=cfg.INPUT.FORMAT,
            ignore_label=ignore_label,
            size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
            dataset_name=dataset_names[0]  # 使用第一个数据集名称
        )


    def __init__(
            self,
            is_train=True,
            *,  # 强制使用关键字参数
            augmentations,
            image_format,
            ignore_label,
            size_divisibility,
            dataset_name
    ):
        super().__init__(
            is_train=is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility
        )

        # 根据数据集名称选择不同的归一化参数，确保是float32类型
        if dataset_name == "remain_dataset_train":
            self.pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            self.pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        elif dataset_name == "learn_dataset_train":
            # voc 数据集 感觉没区别
            # self.pixel_mean = torch.tensor([117.79723406, 113.51956767, 98.64946676], dtype=torch.float32).view(3, 1, 1)
            # self.pixel_std = torch.tensor([58.52638834,58.14435355,59.14101012], dtype=torch.float32).view(3, 1, 1)
            # oxofrd pet数据集
            self.pixel_mean = torch.tensor([119.73433564, 109.83269262, 94.84763627], dtype=torch.float32).view(3, 1, 1)
            self.pixel_std = torch.tensor([57.6707407, 57.19914482, 58.46454528], dtype=torch.float32).view(3, 1, 1)

        elif dataset_name == "forget_dataset_train":
            self.pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            self.pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # 将均值和标准差缩放到0-1范围
        self.pixel_mean = self.pixel_mean / 255.0
        self.pixel_std = self.pixel_std / 255.0

        self.dataset_name = dataset_name

    def __call__(self, dataset_dict):
        # 调用父类的方法进行图像读取和预处理
        dataset_dict = super().__call__(dataset_dict)

        # 确保图像是正确的类型和范围
        if isinstance(dataset_dict["image"], torch.Tensor):
            # 首先转换为float32类型
            image = dataset_dict["image"].float()

            # 如果像素值在0-255范围内，归一化到0-1
            if image.max() > 1.0:
                image = image / 255.0

            # 应用均值和标准差归一化
            dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std

            # 增加调试信息（可选）
            # print(f"Image dtype: {dataset_dict['image'].dtype}")
            # print(f"Image range: [{dataset_dict['image'].min():.4f}, {dataset_dict['image'].max():.4f}]")
            # print(f"Image shape: {dataset_dict['image'].shape}")

        return dataset_dict



