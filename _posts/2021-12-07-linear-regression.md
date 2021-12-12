---
layout: post
title: "Linear Regression: What is it?"
subtitle: "Looking at regression as a mathematician."
background: ""
use_math: true
---

### Overview

Linear regression has been regarded as the basic yet powerful _machine learning_ algorithm. Most beginners start their machine learning journey with this topic. This generally is a simple algorithm to use and understand. It intuitively makes sense, the **loss function**, the **best-fit** line etc. They all seem reasonable at first glance. But, did you ever think why that specific loss function is being used apart from other alternatives? What's there in it to understand mathematically? How will this help us? I'll try to answer all of these questions in this blog. Let's get started!

### Gentle Introduction to Linear Regression

Given a set of datapoints $(x_1, y_1), (x_2, y_2).., (x_n, y_n)$, we need to find a line that is a best representative of these points. How do we do that? That's the whole point of regression right! There are a couple of way to do it, but all of them do the same thing finally, finding the _best-fit_ line (2 dimension), hyperplane (N dimension).

![Datapoint Image](/img/posts/linear-regression/generated_data.svg)

Considering 2D space, we need to find a line of form the $y=mx+c$. Finding this _best-fit_ line is what linear regresion is all about. Wait, wait! what's this _best-fit_ line by the way? Let's look at that.

### Best-fit Line

Simply put, a best-fit line is a line that fits the given data well. Yeah, I know it's pretty explainable from the name "best-fit". For the given data in the plot, the best-fit line is plotted below.

![Best Fit Line](/img/posts/linear-regression/generated_best_fit.svg)

Wow, that fit quite well! How did I come to the conclusion that it fit the data well? Because it minimized my loss function! That takes us to the **loss function**. A best-fit line minimizes the loss function. Let's see what's this loss function is.

### Loss Function

A loss function is a function that tells us how close our predicted values are to the real values.

- A loss function should give 0 when our predictions are same as the original values.
- A loss function should always be positive. Because, the best value we can get is 0, when we we're 100% correct with our predictions so negative values doesn't make sense!

We'll use a loss function called **Mean Squared Error (MSE)**. The formula for the same is as follows:

$$
MSE=L=\frac{\sum _{i=1}^{n}( y_{i} -\hat{y}_{i})^{2}}{n} = \frac{ \begin{array}{l}
\sum _{i=1}^{n}( y_{i}  - \hat{m} x_{i}  - \hat{c})^{2}
\end{array}}{n}
$$

- $y_i$ is the actual value for sample $x_i$
- $\hat{y_i}$ is the predicted values for $x_i$ ( $\hat{ }$ generally means approximate value).

TODO: Add a trivia box saying about the interchange of terms error and loss.

Our **MSE** obeys all the properties of a loss function. We're simply adding all the squares of prediction errors (To keep our function positive) and taking their mean. Now, we need to minimize this function in order to get our best-fit line.

The minimum of **MSE** occurs at 0. But we may never hit 0 in our predictions, this can be evident from the data. The data isn't quite linear, there exists some random error among them. This randomness prevents us from hitting absolute 0! This error is generally represented with $\epsilon$, modifying our original function to $y=mx+c+\epsilon$. So we can never actually approximate $y$. But we can try our best to bring the error close to 0.

TODO: Add a trivia box saying about other loss functions.

### Minimizing the loss function

How do we minimize a function? By differentiation right! Luckily this is also the case here. We need to differentiate our **MSE** and equate it to 0 to get the parameters $m$ and $c$ that gives the minimum value for the function.
Let's find the partial derivatives of **MSE** with respect to $m$ and $c$:

$$
\frac{\partial L }{\partial m} =-2\sum _{i=1}^{n}( y_{i} -\hat{y_i})x_i=-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c})x_i =0
$$

$$
\frac{\partial L }{\partial c} =-2\sum _{i=1}^{n}( y_{i} -\hat{y_i})=-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c}) =0
$$
