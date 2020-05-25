# NNest

Neural network accelerated nested and MCMC sampling. The target distribution is first transformed into a diagonal, unit variance Gaussian by a series of non-linear, invertible, and non-volume preserving flows. Efficient MCMC proposals can then be made in this simpler latent space.

![latent](https://github.com/adammoss/nnest/blob/master/flows.png)

### Installation

NNest can be installed via pip
```
pip install nnest
```
Alternatively the latest version can be obtained by
```
git clone https://github.com/adammoss/nnest
cd nnest
python setup.py install
```

### Nested Sampling

Nested sampling examples can be found in the `examples/nested` directory, and can be run with e.g. 
```
python run.py --x_dim 2 --likelihood rosenbrock
```
There is also an example notebook. Runs can be analysed by
```
python analyse.py
```

### MCMC Sampling

MCMC sampling (work in progress) examples can be found in the `examples/mcmc` directory, and can be run with e.g. 
```
python run.py --x_dim 2 --likelihood rosenbrock
```
There is also an example notebook.

### Attribution

Please cite [Moss (2019)](https://arxiv.org/abs/1903.10860) if you find the 
package useful in your research.