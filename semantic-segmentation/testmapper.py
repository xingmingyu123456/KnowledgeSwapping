import torch
from detectron2.data import DatasetMapper
from detectron2.data.transforms import ResizeShortestEdge, AugmentationList

# 自定义的 DatasetMapper 配置，支持不同的数据集使用不同的归一化参数
class testMapper(DatasetMapper):
    def __init__(self, mean,std,is_train=False, num_workers=4):
        self.mean = mean
        self.std = std
        super().__init__(
            is_train=is_train,
            augmentations=[
                ResizeShortestEdge(
                    short_edge_length=(640, 640),
                    max_size=2560,
                    sample_style="choice"
                ),
            ],
            image_format="RGB",
            instance_mask_format="polygon",
            use_instance_mask=False,
            use_keypoint=False,
            recompute_boxes=False,
        )
        # 新增参数
        self.proposal_topk = None
        self.num_workers = num_workers

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        # 图像已经被转换为 tensor，形状为(C,H,W)
        image = dataset_dict["image"]

        # 确保均值和标准差的维度正确
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)

        # 执行归一化
        normalized_image = (image - mean) / std

        # 更新字典中的图像
        dataset_dict["image"] = normalized_image

        return dataset_dict





