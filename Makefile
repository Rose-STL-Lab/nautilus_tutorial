SHELL := /bin/bash
PATH  := /root/miniconda3/bin:$(PATH)

.PHONY: mnist_hf.py mnist_ddp.py mnist_hvd.py mnist_ds.py cifar10_deepspeed.py

mnist_ds.py:
	deepspeed $@ --deepspeed --deepspeed_config config.json --epochs 10

mnist_hf.py: OMP_NUM_THREADS = 1
mnist_hf.py:
	accelerate launch $@ --epochs 10

mnist_ddp.py: OMP_NUM_THREADS = 1
mnist_ddp.py:
	torchrun --nproc_per_node=8 $@ --batch-size 64 --epochs 10

mnist_hvd.py:
	horovodrun -np 8 -H localhost:8 python $@  --batch-size 64 --epochs 10

cifar10_deepspeed.py:
	deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json
