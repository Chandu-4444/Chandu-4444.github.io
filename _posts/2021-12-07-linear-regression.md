---
layout: post
title: "Linear Regression: A mathematical perspective"
subtitle: "Looking at regression as a mathematician."
background: ""
use_math: true
---

### Overview

The goal of machine learning algorithm is to extract a _relatioinship_ from the data. We need to approximate an unknown true function, _f(**x**)_. The approximate we try to find is $\hat{y} = h(\boldsymbol{X})$, the **hypothesis**. In case of linear regression, $y \in \mathbf{R}$ i.e our target variable is real valued.

### How to do it?

Let's consider the following plot:

![Plot Image](/img/posts/linear-regresison/plot.png)

The red points $(x_1, y_1), (x_2, y_2) .., (x_n, y_n)$ are the samples from the true distribution (f) that we need to approximate. The very basic intuition we can come up with is something like $y = mx+c$. Yes, this is absolutely correct! Infact, the data I've generated above is based on that equation. We need to find $m$ and $c$ now. You might have learned the technique of deriving the equation of a line from two points, given that the two points lie on the line. This is good if the data looks exactly like a straight line. But look at the plot above, the points aren't very linear, I introduced some randomness in generating the data ($y = mx + c +\epsilon$). That's the reason they aren't so linear. This is often the case in data we get in real world cases. There will be some noise, some inconsistency, missing values etc among the data. The prime objective is to approximate the true function as close as possible, we might not need to be perfect. To be honest, we'll never be able to find the exact true function! Owing to the randomness in the generated data. So, we'll ignore the $\epsilon$ and pretend that it's never there.

Let's get back to the question, finding $m$ and $c$. To match with conventional terms, let's make our $m = \beta_1$ and $c = \beta_0$. So our initial approximate funtion becomes $\hat{y} = \beta_1x + \beta_0$. Take note of that little hat ( $\hat{}$ ) above y, that indicates that the y is an approximate. Let's move on to the next section, where we look at our objective function.

### Least-Squares Estimation of the parameters

The parameters $\beta_1$ and $\beta_0$ (**Regression Coefficients**) are unknown, and must be estimated from the data samples given to us $(x_1, y_1), (x_2, y_2) .., (x_n, y_n)$. The **method of least squares** is used to estimate $\beta_0$ and $\beta_1$. We are only interested in the difference of the actual $y_i$ and the predicted $\hat{y}_i$, that's the we're squaring the difference. The total error is sum of each $(y_i - \hat{y}_i)$.

The first equation is called **population regression model** while the second is called **sample regresion model**, written in terms of the _n_ pairs of data. Using this latter notation, we formulate our least squares as follows:

There's actually a nice mathematical intuition on why least squares is the suitable objective function for linear regression that uses **maximum likelihood estimation** (I'll write about that in another post).

Our problem boiled down to the above equation, we need to find $\beta_1$ and $\beta_0$ such that $\mathbf{L}$ will get minimised. What's that technique used to find minimum of a function? Differentiation ofcourse! We'll find the partial derivatives of $\mathbf{L}$ with respect to $\beta_1$ and $\beta_0$ and equate them to zero.

```python
import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")
```
