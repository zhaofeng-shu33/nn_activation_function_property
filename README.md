# Principle
Suppose $`X_1, X_2, \dots,X_k`$ and $`Y`$ are uniformly distributed random vectors on the n-dimensional unit sphere.
These random vectors are indepently distributed.
We try to compare the minimum value of
```math
E[\min || Y - \sum_{i=1}^k w_i X_i ||^2]
```
with
```math
E[\min || Y - \sigma(\sum_{i=1}^{k-1} w_i X_i) - b ||^2]
```
where $`\sigma`$ is a non-linear function and $`w_i`$ is random variables depedent on $`X_i, Y`$.

# How to
```shell
python optimization.py --activate tf.sigmoid
```

We found:
1. If the activation function is nonnegative, the result is worst even with a constant bias.
1. `tf.tanh` is a little bit better than the case with no activation function.


