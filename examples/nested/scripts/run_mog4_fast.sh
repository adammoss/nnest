#!/bin/bash
for i in `seq 1 1`
do
	python run.py --x_dim 3 --num_slow 2 --likelihood mixture
	python run.py --x_dim 4 --num_slow 2 --likelihood mixture
	python run.py --x_dim 5 --num_slow 2 --likelihood mixture
	python run.py --x_dim 10 --num_slow 2 --likelihood mixture
	python run.py --x_dim 20 --num_slow 2 --likelihood mixture
	python run.py --x_dim 30 --num_slow 2 --likelihood mixture
done