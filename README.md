# Convergence of Riemannian Stochastic Gradient Descent on Hadamard Manifolds
Code for reproducing experiments in our paper.

## Abstract
Novel convergence analyses are presented of
Riemannian stochastic gradient descent (RSGD) on a Hadamard manifold.
RSGD is the most basic Riemannian stochastic optimization algorithm and
is used in many applications in the field of machine learning.
The analyses incorporate the concept of mini-batch learning used in deep learning and
overcome several problems in previous analyses.
Four types of convergence analysis are described for both constant and decreasing step sizes.
The number of steps needed for RSGD convergence is shown to be
a convex monotone decreasing function of the batch size.
Application of RSGD with several batch sizes to
a Riemannian stochastic optimization problem on a symmetric positive
definite manifold theoretically shows that increasing the batch size improves RSGD performance.
Numerical evaluation of the relationship between
batch size and RSGD performance provides evidence supporting the theoretical results.

## Dependencies
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/stable/index.html)

## Usage
### Experiment
```Python
python src/main.py
```
### Plot
```Python
python src/show.py
```
