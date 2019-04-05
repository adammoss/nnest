# NNest

Neural network accelerated nested and MCMC sampling. Nested sampling examples can be found in the `examples/nested` directory, and can be run with e.g. 
```
python examples/nested/rosenbrock.py --x_dim 2
```
MCMC sampling examples can be found in the `examples/mcmc` directory, and can be run with e.g. 
```
python examples/mcmc/rosenbrock.py --x_dim 2
```
Runs can be analysed by e.g.
```
python analyse.py --name rosenbrock --x_dim 2 -plot
python analyse.py --name rosenbrock_mcmc --x_dim 2 --sampler mcmc -plot
```
