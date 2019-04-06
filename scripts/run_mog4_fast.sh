#!/bin/bash
for i in `seq 1 1`
do
	python examples/nested/mog4_fast.py --x_dim 3
	python examples/nested/mog4_fast.py --x_dim 4
	python examples/nested/mog4_fast.py --x_dim 5
	python examples/nested/mog4_fast.py --x_dim 10
	python examples/nested/mog4_fast.py --x_dim 20
	python examples/nested/mog4_fast.py --x_dim 30
done