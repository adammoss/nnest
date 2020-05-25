#!/bin/bash
python run.py --switch 0.02 --test_samples 10000 --test_mcmc_steps 30 --train_iters 100 --likelihood rosenbrock
python run.py --switch 0.02 --test_samples 10000 --test_mcmc_steps 30 --train_iters 100 --likelihood himmelblau