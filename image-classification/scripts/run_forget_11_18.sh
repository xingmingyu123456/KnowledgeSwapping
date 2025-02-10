export CUDA_VISIBLE_DEVICES=4
#NUM_FIRST_CLS=80
#PER_FORGET_CLS=$((100-$NUM_FIRST_CLS))

# # GS-LoRA
# for lr in 1e-2
# do
# for beta in 0.15
# do
# python3 -u train/exchange_order_bird_simple_11_18.py -b 80 -w 0 -d casia100 -n VIT -e 5 \
#     -head Softmax --outdir /home/xmy/code/papercodegithub/image-classification/log/test \
#     --warmup-epochs 0 --lr $lr --num_workers 8  --lora_rank 8 --decay-epochs 100 \
#     --vit_depth 12 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
#     -r /home/xmy/code/imagenet100_small_checkpoint_2024-09-27-00-31-30.bin \
#     --BND 105 --beta $beta --alpha 0.005 --min-lr 1e-5 --num_tasks 1 --wandb_group forget_cl_new \
#     --cl_beta_list 0.2 0.25 0.25 0.25
# done
# done

 python3 -u train/exchange_fltf.py -b 80 -w 0 -d casia100 -n VIT -e 5 \
     -head Softmax --outdir /home/xmy/code/classify-bird/result/cub_learn_5_fftl \
     --warmup-epochs 0 --lr 1e-2   --num_workers 8  --lora_rank 8 --decay-epochs 100 \
     --vit_depth 12 \
     -r /home/xmy/code/imagenet100_small_checkpoint_2024-09-27-00-31-30.bin \
     --BND 105 --beta 0.2 --alpha 0.005 --min-lr 1e-5 --wandb_group swapping