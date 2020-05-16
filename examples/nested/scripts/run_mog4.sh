#!/bin/bash
for i in `seq 1 1`
do
	python mog4.py --x_dim 2
	python mog4.py --x_dim 3
	python mog4.py --x_dim 4
	python mog4.py --x_dim 5
	python mog4.py --x_dim 10
	python mog4.py --x_dim 20
	python mog4.py --x_dim 30
done