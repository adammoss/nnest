#!/bin/bash
for i in `seq 1 1`
do
	python run.py --x_dim 2 --likelihood mixture
	python run.py --x_dim 3 --likelihood mixture
	python run.py --x_dim 4 --likelihood mixture
	python run.py --x_dim 5 --likelihood mixture
	python run.py --x_dim 10 --likelihood mixture
	python run.py --x_dim 20 --likelihood mixture
	python run.py --x_dim 30 --likelihood mixture
done