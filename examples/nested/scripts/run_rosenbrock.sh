#!/bin/bash
for i in `seq 1 1`
do
	python run.py --x_dim 2 --likelihood rosenbrock
	python run.py --x_dim 3 --likelihood rosenbrock
	python run.py --x_dim 4 --likelihood rosenbrock
	python run.py --x_dim 5 --likelihood rosenbrock
	python run.py --x_dim 10 --likelihood rosenbrock
	python run.py --x_dim 20 --likelihood rosenbrock
	python run.py --x_dim 30 --likelihood rosenbrock
done