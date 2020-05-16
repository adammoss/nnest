#!/bin/bash
for i in `seq 1 1`
do
	python rosenbrock.py --x_dim 2
	python rosenbrock.py --x_dim 3
	python rosenbrock.py --x_dim 4
	python rosenbrock.py --x_dim 5
	python rosenbrock.py --x_dim 10
	python rosenbrock.py --x_dim 20
	python rosenbrock.py --x_dim 30
done