(CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar10 --num_classes 10 --corruption_type flip2 --corruption_ratio 0.2 >10_02.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar10 --num_classes 10 --corruption_type flip2 --corruption_ratio 0.4 >10_04.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar100 --num_classes 100 --corruption_type flip2 --corruption_ratio 0.2 >100_02.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python MetaIADA.py --dataset cifar100 --num_classes 100 --corruption_type flip2 --corruption_ratio 0.4 >100_04.txt 2>&1 \
) &