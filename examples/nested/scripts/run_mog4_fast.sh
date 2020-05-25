#!/bin/bash
for i in `seq 1 1`
do
	python run.py --x_dim 3 --nslow 2 --likelihood mixture
	python run.py --x_dim 4 --nslow 2 --likelihood mixture
	python run.py --x_dim 5 --nslow 2 --likelihood mixture
	python run.py --x_dim 10 --nslow 2 --likelihood mixture
	python run.py --x_dim 20 --nslow 2 --likelihood mixture
	python run.py --x_dim 30 --nslow 2 --likelihood mixture
done