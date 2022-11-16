SHELL := /bin/bash
PATH  := /root/miniconda3/bin:$(PATH)

.PHONY: mnist_ddp.py mnist.py mnist_hvd.py

mnist.py:
	torchrun $@

mnist_ddp.py: OMP_NUM_THREADS = 1
mnist_ddp.py:
	torchrun --nproc_per_node=8 $@ --batch-size 64 --epochs 14

mnist_hvd.py:
	horovodrun -np 8 -H localhost:8 python $@  --batch-size 64