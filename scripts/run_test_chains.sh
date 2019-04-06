#!/bin/bash
python examples/nested/rosenbrock.py --switch 0.02 --test_samples 10000 --test_mcmc_steps 30 --train_iters 100
python examples/nested/himmelblau.py --switch 0.02 --test_samples 10000 --test_mcmc_steps 30 --train_iters 100