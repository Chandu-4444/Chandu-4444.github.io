---
layout: post
title: "Linear Regression: What is it?"
subtitle: "Looking at regression as a mathematician."
background: ""
use_math: true
comments: true
toc: true
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

We'll use a loss function called **Sum of Squared Error (SSE)**. The formula for the same is as follows:

$$
SSE=L=\sum _{i=1}^{n}( y_{i} -\hat{y}_{i})^{2} = \sum _{i=1}^{n}( y_{i}  - \hat{m} x_{i}  - \hat{c})^{2}
$$

- $y_i$ is the actual value for sample $x_i$
- $\hat{y_i}$ is the predicted values for $x_i$ ( $\hat{ }$ generally means approximate value).

<div class="todo-container">
<span class="todo-icon">💡</span> <span class="todo-head">Loss VS Error, Do we care? </span>
<div class="todo-body">
I intentionally used the term <i>loss</i> initially just to make you get the essence of the similarity between <i>loss</i> and <i>error</i>. These are basically the same. If the error is high, our loss is also high (Think of it as a game of bow and arrow).
</div>
</div>

Our **SSE** obeys all the properties of a loss function. We're simply adding all the squares of prediction errors (To keep our function positive) and taking their mean. Now, we need to minimize this function in order to get our best-fit line.

The minimum of **SSE** occurs at 0. But we may never hit 0 in our predictions, this can be evident from the data. The data isn't quite linear, there exists some random error among them. This randomness prevents us from hitting absolute 0! This error is generally represented with $\epsilon$, modifying our original function to $y=mx+c+\epsilon$. So we can never actually approximate $y$. But we can try our best to bring the error close to 0.

### Minimizing the loss function

How do we minimize a function? By differentiation right! Luckily this is also the case here. We need to differentiate our **SSE** and equate it to 0 to get the parameters $m$ and $c$ that gives the minimum value for the function.
Let's find the partial derivatives of **SSE** with respect to $m$ and $c$:

$$
\frac{\partial L }{\partial \hat{m}} =-2\sum _{i=1}^{n}( y_{i} -\hat{y_i})x_i=-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c})x_i =0
$$

$$
\frac{\partial L }{\partial \hat{c}} =-2\sum _{i=1}^{n}( y_{i} -\hat{y_i})=-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c}) =0
$$

Simplying the second equation (to make the process easier),

$$
\begin{split}
\frac{\partial L }{\partial \hat{c}} & =-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c}) =0
\\
&\implies -2(\sum _{i=1}^{n}y_i-\sum _{i=1}^{n}\hat{m}x_i-\sum _{i=1}^{n}\hat{c})=0\\
&\implies-2(\sum _{i=1}^{n}y_i-\hat{m}\sum _{i=1}^{n}x_i-\hat{c}\sum _{i=1}^{n}1)=0\\
& \implies \sum _{i=1}^{n}y_i-\hat{m}\sum _{i=1}^{n}x_i-\hat{c}n=0\\
& \implies \sum _{i=1}^{n}y_i-\hat{m}\sum _{i=1}^{n}x_i=\hat{c}n\\
& \implies\hat{c} = \frac{\sum_{i=1}^ny_i}{n} - \hat{m}\frac{\sum_{i=1}^nx_i}{n}\\
& \implies \boxed{\hat{c}=\bar{y}-\hat{m}\bar{x}}
\end{split}
\\
$$

We now need the value of $\hat{m}$, We'll solve this by using the value of $\hat{c}$ we obtained above:

$$
\begin{split}
\frac{\partial L}{\partial \hat{m}} & = -2\sum_{i=1}^n(y_i-\hat{m}x_i-\hat{c})x_i=0\\
& \implies \sum_{i=1}^ny_ix_i-\hat{m}\sum_{i=1}^nx_i^2-\hat{c}\sum_{i=1}^nx_i=0\\
& \implies \hat{m}\sum_{i=1}^nx_i^2=\sum_{i=1}^ny_ix_i-\hat{c}\sum_{i=1}^nx_i\\
& \text{Using the above value, } \hat{c}=\bar{y}-\hat{m}\bar{x}\\
& \implies \hat{m}\sum_{i=1}^nx_i^2=\sum_{i=1}^ny_ix_i-\bigg[\frac{1}{n}\sum_{i=1}^ny_i - \hat{m}\frac{1}{n}\sum_{i=1}^nx_i\bigg]\sum_{i=1}^nx_i=0\\
& \implies \hat{m}\sum_{i=1}^nx_i^2=\sum_{i=1}^ny_ix_i - \frac{1}{n}\sum_{i=1}^ny_i\sum_{i=1}^nx_i+\frac{\hat{m}}{n}(\sum_{i=1}^nx_i)^2=0\\
& \implies \hat{m}\sum_{i=1}^nx_i^2-\frac{\hat{m}}{n}(\sum_{i=1}^nx_i)^2=\sum_{i=1}^ny_ix_i - \frac{1}{n}\sum_{i=1}^ny_i\sum_{i=1}^nx_i\\
& \implies \hat{m}\bigg[\sum_{i=1}^nx_i^2- \frac{(\sum_{i=1}^nx_i)^2}{n}  \bigg] = \sum_{i=1}^ny_ix_i - \frac{\sum_{i=1}^ny_i\sum_{i=1}^nx_i}{n}\\
& \implies \boxed{\hat{m} = \frac{\sum_{i=1}^ny_ix_i - \frac{1}{n}\sum_{i=1}^ny_i\sum_{i=1}^nx_i}{ \bigg[\sum_{i=1}^nx_i^2- \frac{1}{n}(\sum_{i=1}^nx_i)^2\bigg]}}
\end{split}
$$

Uff, that took sometime with $\LaTeX$. Hopefully, I did everything without any mistakes. It looks like I did! Coming back to the equations, we derived the solutions to find $\hat{m}$ and $\hat{c}$ this values gives the minimum value for our **SSE**. This is the basic structure of the **Ordinary Least Squares (OLS)** and this type of regression is called **Ordinary Least Squares Regression** (the name is pretty self explanatory).

This works well for one _independent variable_ ($$x$$), but what if we have more? This is where we have to deal with vectors and matrices! Till now we've dealt with scalars, in the next section we'll derive the solutions to **SSE** using **OLS** for matrices. Buckle up for some matrix algebra!

### OLS for matrices

This kind of regression comes under **multiple regresion** where we have multiple input features with one real valued output. Our $$x_i \in R^k$$ i.e any point $$x_i$$ can be represented as $$(x^1, x^2..,x^k)$$.

We need to change the notation here, from now on :

- $$\mathbf{X}_{r\times c}$$ (bold uppercase letter) represents a matrix with $$r$$ rows and $$c$$ columns.
- $$\mathbf{x}$$ (bold lowercase letter) represents a vector.
- $$x$$ (lowercase letter) represents a scalar.

I'll mention the dimension for the matrix where ever possible, cause this had been the cause for most of my _confusion_. I'll first walkthrought how we represent inputs in multiple regression.

$$
\mathbf{x}_i = [x^{1}_i, x^{2}_i.., x^{k}_i]\\
\small{\text{ (Please note that these superscripts are not powers, they are dimensions) }}\\
$$

Each input vector has $k$ features.

<div class="todo-container">
<span class="todo-icon">💡</span> <span class="todo-head">What are features btw?</span>

<div class="todo-body">
Features are any attributes that might be useful in predicting our output variable. For example, <i>Plot Area</i>, <i>Neighbourhood</i>, <i>Number of Floors</i> etc might be the features when predicting the <i>Price of a house</i>
</div>
</div>

Now our Matrix $$\mathbf{X}_{n\times k}$$ contains $$\mathbf{x}_i$$ ($$1 \times k$$ dimension) as its rows

$$
  \mathbf{X}_{n\times k} =
  \left[ {\begin{array}{cc}
    \mathbf{x}_{1} \\
    \mathbf{x}_{2} \\
    .\\
    .\\
    \mathbf{x}_{n}
  \end{array} } \right] =
  \left[ {\begin{array}{cc}
          x_1^1 & x_1^2 & .., & x_1^k \\
          x_2^1 & x_2^2 & .., & x_2^k \\
          . & . & .., & . \\
          x_n^1 & x_n^2 & .., & x_n^k\\
      \end{array}} \right]
$$

Now our equation becomes,

$$
\mathbf{\hat{y}} \text{ }(n \times 1 \text{ dimension}) = \mathbf{X}_{n \times k}\mathbf{W}_{k \times 1}
$$

This is an equation of a **hyperplane**. Now we need to introduce the _intercept_ term (Similar to $$\hat{c}$$ in $$\hat{y}=\hat{m}x+\hat{c}$$). For this, I think it's better to start with 2D again.

We'll try to vectorize the 2D equation:

$$
\hat{y} = \left[ {\begin{array}{cc} x \text{ } 1 \end{array} } \right] \left[ {\begin{array}{cc}
    \hat{m} \\
    \hat{c} \\
  \end{array} } \right] = \hat{m}x + 1 \cdot \hat{c}
$$

Observe the $$1$$ appended to the input vector. This acommodates for the intercept term. Using this intuition, we'll now add the intercept term to our equation. Our input matrix $$\mathbf{X}$$ becomes:

$$
  \mathbf{X}_{n\times (k+1)} =
  \left[ {\begin{array}{cc}
    \mathbf{x}_{1} & 1 \\
    \mathbf{x}_{2} & 1 \\
    .\\
    .\\
    \mathbf{x}_{n} & 1
  \end{array} } \right] =
  \left[ {\begin{array}{cc}
          x_1^1 & x_1^2 & .., & x_1^k & 1 \\
          x_2^1 & x_2^2 & .., & x_2^k & 1\\
          . & . & .., & . \\
          x_n^1 & x_n^2 & .., & x_n^k & 1\\
      \end{array}} \right]
$$

Our _weight_ (or slope) vector $$\mathbf{W}$$ becomes:

$$
\mathbf{W}_{(k+1) \times 1} = \left[ {\begin{array}{cc}
    w_1 \\
    w_2 \\
    . \\
    . \\
    w_k\\
    w_0
  \end{array} } \right]
$$

For historic reasons (not really), we'll now refer to our interept term with $$w_0$$. Finally rewriting our equation:

$$
\mathbf{\hat{y}} \text{ }(n \times 1 \text{ dimension}) = \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1}
$$

Ok, we've formulated our equation. Next comes the loss function, which will be the same **Sum of Squared Error (SSE)** we've seen earlier:

$$
\text{Error in prediction} = \mathbf{y} - \mathbf{\hat{y}} \\
$$

We need to square this to avoid negative error:

$$
{\begin{split}
    \text{SSE} & = (\mathbf{y} - \mathbf{\hat{y}})^2 = (\mathbf{y} - \mathbf{\hat{y}})^T(\mathbf{y} - \mathbf{\hat{y}}) = (\mathbf{y} - \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1})^T(\mathbf{y} - \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1}) \text{ } & [\text{Using the result } \mathbf{X}^2 = \mathbf{X}^T\mathbf{X}] \\

    & \implies (\mathbf{y}^T - \mathbf{W}_{1 \times (k+1)}^T\mathbf{X}_{(k+1) \times n}^T)(\mathbf{y} - \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1}) \text{ } & [\text{Using the result } (\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T] \\

    & \implies \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X}\mathbf{W} - \mathbf{W}^T\mathbf{X}^T\mathbf{y} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W} \text{ } & [\text{I won't mention dimensions anymore } ] \\
\end{split} }
$$

Now, $$\mathbf{y}^T\mathbf{X}\mathbf{W}$$ and $$\mathbf{W}^T\mathbf{X}^T\mathbf{y}$$ produces the same result (feel free to try it on few examples).

$$
\text{SSE} = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{W} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}
$$

The next step is to take the derivative of SSE with respect to matrix $$\mathbf{W}$$. Since we're dealing with set of variables (the constant and regression coefficients) we again use partial differentiation:

$$
\begin{split}
  \frac{\partial SSE}{\partial \mathbf{W}} & = \frac{\partial }{\partial \mathbf{W}} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{W} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}) \\
  & \implies \frac{\partial}{\partial \mathbf{W}}(\mathbf{y}^T\mathbf{y}) - \frac{\partial}{\partial \mathbf{W}}(2\mathbf{y}^T\mathbf{X}\mathbf{W}) + \frac{\partial}{\partial \mathbf{W}}(\mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}) \\
\end{split}
$$

This looks like something new if you don't know calculus on matrices. I'll try to map this with linear context.

The first term $$\mathbf{y}^T\mathbf{y}$$ is a constant when differentiating with respect to $$\mathbf{W}$$. So this will be $$0$$.

The second term $$2\mathbf{y}^T\mathbf{X}\mathbf{W}$$ this is a "linear form" in $$\mathbf{W}$$ and is equivalent to scalar term in which the variable we are differentiating with respect to is raised to the first power. This means that we can get the derivative by dropping the $$\mathbf{W}$$. Since this is matrix calculus, we need to takt the transpose of whatever's remaining, giving us $$-2\mathbf{X}^T\mathbf{y}$$.

The final term $$\mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}$$, is a "quadratic form" and is equivalent to scalar term in which the variable we are differentiating with respect to is raised to the second power. This indicates that we can get the derivative by dropping $$\mathbf{W}$$ and then taking the transpose, finally multiplying this by 2 giving us $$2\mathbf{X}^T\mathbf{X}\mathbf{W}$$.

$$
\begin{split}
  \frac{\partial SSE}{\partial \mathbf{W}} & = \frac{\partial}{\partial \mathbf{W}}(\mathbf{y}^T\mathbf{y}) - \frac{\partial}{\partial \mathbf{W}}(2\mathbf{y}^T\mathbf{X}\mathbf{W}) + \frac{\partial}{\partial \mathbf{W}}(\mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}) \\
  & \implies 0 - 2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{W} \\
  & \text{Equating this to 0 as we did for linear case previously,} \\
  & \implies - 2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{W} = 0 \\
  & \implies 2\mathbf{X}^T\mathbf{X}\mathbf{W} = 2\mathbf{X}^T\mathbf{y} \\
  & \implies \mathbf{X}^T\mathbf{X}\mathbf{W} = \mathbf{X}^T\mathbf{y} \\
  & \text{Multiplying } (\mathbf{X}^T\mathbf{X})^{-1} \text{ on both sides,} \\
  & \implies (\mathbf{X}^T\mathbf{X})^{-1} (\mathbf{X}^T\mathbf{X})\mathbf{W} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y} \\
  & \implies \mathbf{I}\mathbf{W} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y} \\
  & \implies  \boxed{\mathbf{W} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}} \\
\end{split}
$$

Now, that gives us the corresponding _weights_ that can be used to construct a hyperplane that best fits our data $$\mathbf{X}$$. This equation is called **Normal Equation**.

That's the mathematical reason behind the first ML algorithm you've probably learnt! Stay tuned for more :) Also, do let me know in the comments if you find any error or any suggestion.

{% include disqus.html %}
