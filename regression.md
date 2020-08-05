## Linear Regression

$Y \sim N(w^Tx, \sigma^2)$. $\sigma^2$ is constant for each sample. The ML leads to square loss function.

## Logistic Regression

The variable $y$ only takes two values $0,1$.  $Y\sim Bern(h(w^Tx))$

The Logistic Regression fits a probability function $p=P(Y=1 | X, w)= \frac{1}{1+\exp(- w^T x)}$. 

To fit the model, instead of mean square error loss, usually we use the cross-entropy loss, which is equivalent to the maximal likelihood solution:
$$
L(w | y; x) = \sum_{i=1}^n y_i \log h_w(x_i) + (1-y_i) \log (1-h_w(x_i))
$$
where $h_w(x) = \frac{1}{1+\exp(-w^Tx)}$.

Besides using SGD to minimize the loss, we can use Iteratively Reweighted Least Sqaures (IRLS) to get the optimal weight.

## Poisson Regression

The response variable $y$ only takes positive integer values, $0,1, 2, \dots$

The mean $\lambda = e^{w^T x}$. Using ML, the Loss function is
$$
L(w | y;x) = \sum_{i=1}^n (y_i w^Tx - \exp(w^Tx))
$$
