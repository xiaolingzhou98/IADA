(CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar10 --num_classes 10 --imb_factor 10 >10_10.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar10 --num_classes 10 --imb_factor 100 >10_100.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar100 --num_classes 100 --imb_factor 10 >100_10.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar100 --num_classes 100 --imb_factor 100 >100_100.txt 2>&1 \
) &