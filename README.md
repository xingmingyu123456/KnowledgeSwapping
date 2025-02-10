# KnowledgeSwapping
This is the official code implementation of Knowledge Swapping via Learning and Unlearning. 
Knowledge Swapping is a novel
task designed to selectively regulate knowledge
of a pretrained model by enabling the forgetting
of user-specified information, retaining essential
knowledge, and acquiring new knowledge simultaneously.
![示意图](./img/img.png)

# Getting Started
## Installation
Download the repo:
```angular2html
git clone https://github.com/xingmingyu123456/KnowledgeSwapping.git
cd KnowledgeSwapping
```
Install needed packages:
```angular2html
conda create -n swapping python=3.12
conda activate swapping
pip install -r requirements.txt
```
Compiling CUDA operators
```angular2html
cd object-dection/models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```
## Dataset
### Image Classification
Download the dataset from [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and put them in the `data` folder.
### Object Detection
Download the dataset from [COCO](https://cocodataset.org/#download) and put it in the `data` folder.
### Semantic Segmentation
Download the dataset from [COCO](https://cocodataset.org/#download) and put it in the `data` folder.


## Pretrain model


## Image Classification
first learn then forget
```angular2html
cd image-classification
python3 -u train/exchange_fltf.py -b 80 -w 0 -d casia100 -n VIT -e 5 \  
    -head Softmax --outdir /home/xmy/code/classify-bird/result/cub_learn_5_fftl \  
    --warmup-epochs 0 --lr 1e-2   --num_workers 8  --lora_rank 8 --decay-epochs 100 \  
    --vit_depth 12 -r /home/xmy/code/imagenet100_small_checkpoint_2024-09-27-00-31-30.bin \  
    --BND 105 --beta 0.2 --alpha 0.005 --min-lr 1e-5 --wandb_group swapping
```
first forget then learn
```angular2html
cd image-classification
python3 -u train/exchange_fftl.py -b 80 -w 0 -d casia100 -n VIT -e 5 \  
   -head Softmax --outdir /home/xmy/code/classify-bird/result/cub_learn_5_fftl \  
   --warmup-epochs 0 --lr 1e-2  --num_workers 8  --lora_rank 8 --decay-epochs 100 \  
   --vit_depth 12 -r /home/xmy/code/imagenet100_small_checkpoint_2024-09-27-00-31-30.bin \  
   --BND 105 --beta 0.2 --alpha 0.005 --min-lr 1e-5 --wandb_group swapping
```
## Object Detection
first learn then forget
```angular2html
cd object-detection
CUDA_VISIBLE_DEVICES=5,7 torchrun --nproc_per_node=2 swapping_fltf_ddp.py --pretrain_model_path /data1/xmy/DINO/logs/swin4c01-14-14-47-40-cl101/checkpoint0004.pth -c ./config/DINO/DINO_4scale_swin_swap.py --lora_rank 8 --lora_pos="FFN" --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --forget_path /data1/xmy/DINO/data/coco0120/cocobench_forget --learn_path /data1/xmy/DINO/data/doglearn --remain_path /data1/xmy/DINO/data/coco0120/cocobench_remain --output_dir /data1/xmy/DINO/logs/swin4c --save_log --find_unused_params
```
first forget then learn
```angular2html
cd object-detection
CUDA_VISIBLE_DEVICES=5,7 torchrun --nproc_per_node=2 swapping_fftl_ddp.py --pretrain_model_path /data1/xmy/DINO/logs/swin4c01-14-14-47-40-cl101/checkpoint0004.pth -c ./config/DINO/DINO_4scale_swin_swap.py --lora_rank 8 --lora_pos="FFN" --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --forget_path /data1/xmy/DINO/data/coco0120/cocobench_forget --learn_path /data1/xmy/DINO/data/doglearn --remain_path /data1/xmy/DINO/data/coco0120/cocobench_remain --output_dir /data1/xmy/DINO/logs/swin4c --save_log --find_unused_params
```
## Semantic Segmentation
first learn then forget
```angular2html
cd semantic-segmentation
CUDA_VISIBLE_DEVICES=3,4 python swapping_fltf.py  --num-gpus 2 --config-file configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_swap.yaml --learnsetname voc --learnset /data1/xmy/Mask2Former-main/voclearn --forgetset /data1/xmy/forgetsmall --retainset /data1/xmy/remainsmall  MODEL.WEIGHTS /home/xmy/Mask2Former-250/exp/log/11-23-18-53/model_0002999.pth SOLVER.IMS_PER_BATCH 6 OUTPUT_DIR exp/log 
```
first forget then learn
```angular2html
cd semantic-segmentation
CUDA_VISIBLE_DEVICES=6,7 python swapping_fltf.py  --num-gpus 2 --config-file configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_swap.yaml --learnsetname voc --learnset /data1/xmy/Mask2Former-main/data/voclearn --forgetset /home/xmy/code/Mask2Former-main/datasets/adebased/forgetsmall --retainset /home/xmy/code/Mask2Former-main/datasets/adebased/remainsmall  MODEL.WEIGHTS /data1/xmy/Mask2Former-main/exp/log/11-23-18-53/model_0002999.pth SOLVER.IMS_PER_BATCH 6 OUTPUT_DIR exp/log 
```

## Acknowledgments
The code is partially from the below repos.

- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [DINO](https://github.com/IDEA-Research/DINO)
- [GS-LoRA](https://github.com/bjzhb666/GS-LoRA)

Please follow their licenses. Thanks for their awesome works.
