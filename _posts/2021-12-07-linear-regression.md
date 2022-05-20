---
layout: post
title: "Linear Regression: What is it?"
subtitle: "Looking at regression as a mathematician."
background: ""
categories: theory maths
# use_math: true
---

Linear regression has been regarded as the basic yet powerful _machine learning_ algorithm. Most beginners start their machine learning journey with this topic. This generally is a simple algorithm to use and understand. It intuitively makes sense, the **loss function**, the **best-fit** line etc. They all seem reasonable at first glance. But, did you ever think why that specific loss function is being used apart from other alternatives? What's there in it to understand mathematically? How will this help us? I'll try to answer all of these questions in this blog. Let's get started!

### Gentle Introduction to Linear Regression

Given a set of datapoints <!-- $(x_1, y_1), (x_2, y_2).., (x_n, y_n)$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/fUUVUFvor7.svg">, we need to find a line that is a best representative of these points. How do we do that? That's the whole point of regression right! There are a couple of way to do it, but all of them do the same thing finally, finding the _best-fit_ line (2 dimension), hyperplane (N dimension).

![Datapoint Image](/img/posts/linear-regression/generated_data.svg)

Considering 2D space, we need to find a line of form the <!-- $y=mx+c$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/NYLSPX0fi5.svg">. Finding this _best-fit_ line is what linear regresion is all about. Wait, wait! what's this _best-fit_ line by the way? Let's look at that.

### Best-fit Line

Simply put, a best-fit line is a line that fits the given data well. Yeah, I know it's pretty explainable from the name "best-fit". For the given data in the plot, the best-fit line is plotted below.

![Best Fit Line](/img/posts/linear-regression/generated_best_fit.svg)

Wow, that fit quite well! How did I come to the conclusion that it fit the data well? Because it minimized my loss function! That takes us to the **loss function**. A best-fit line minimizes the loss function. Let's see what's this loss function is.

### Loss Function

A loss function is a function that tells us how close our predicted values are to the real values.

- A loss function should give 0 when our predictions are same as the original values.
- A loss function should always be positive. Because, the best value we can get is 0, when we we're 100% correct with our predictions so negative values doesn't make sense!

We'll use a loss function called **Sum of Squared Error (SSE)**. The formula for the same is as follows:

<!-- $
SSE=L=\sum _{i=1}^{n}( y_{i} -\hat{y}_{i})^{2} = \sum _{i=1}^{n}( y_{i}  - \hat{m} x_{i}  - \hat{c})^{2}
$ --> <img style="transform: translateY(0.1em); background: white; padding: 20px;" src="/svg/FBrOZUTVzQ.svg">

- <!-- $y_i$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/o60nv0ShX3.svg"> is the actual value for sample <!-- $x_i$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/I1ACbwrrhu.svg">
- <!-- $\hat{y_i}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/XRAhUlmt7m.svg"> is the predicted values for <!-- $x_i$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/eYxNCPiUND.svg"> ( <!-- $\hat{ }$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/8ZhCV3OUV3.svg"> generally means approximate value).

Our **SSE** obeys all the properties of a loss function. We're simply adding all the squares of prediction errors (To keep our function positive). Squaring allows us to give more weightage to large errors and less weightage to small errors. For example, if we predict <!-- $9.5$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/PGdI1zdDFw.svg"> when the actual value is <!-- $9.4$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/oZx3u00Mqt.svg"> the error turns out to be <!-- $0.1$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/E2X8dhLvrW.svg"> and its square is <!-- $0.01$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/M8sbc5eBDQ.svg">. If we predict 9 when the actual value is <!-- $7$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/psP1yUePCY.svg">, the error is <!-- $2$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/GdqbcRoOr3.svg"> and the square is <!-- $4$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/8e8YeZgR9J.svg">. This way we're handling the errors appropriately. Now, we need to minimize this function in order to get our best-fit line.

The minimum of **SSE** occurs at 0. But we may never hit 0 in our predictions, this can be evident from the data. The data isn't quite linear, there exists some random error among them. This randomness prevents us from hitting absolute 0! This error is generally represented with <!-- $\epsilon$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/rHFGPm3NOm.svg">, modifying our original function to <!-- $y=mx+c+\epsilon$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/XS176SzpvG.svg">. So we can never actually approximate <!-- $y$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/1acAJ5KJv2.svg">. But we can try our best to bring the error close to 0.

### Minimizing the loss function

How do we minimize a function? By differentiation right! Luckily this is also the case here. We need to differentiate our **SSE** and equate it to 0 to get the parameters $m$ and $c$ that gives the minimum value for the function.
Let's find the partial derivatives of **SSE** with respect to $m$ and $c$:

<!-- $$
\frac{\partial L }{\partial \hat{m}} =-2\sum _{i=1}^{n}( y_{i} -\hat{y_i})x_i=-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c})x_i =0
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/XlSNsavCM7.svg"></div>

<!-- $$
\frac{\partial L }{\partial \hat{c}} =-2\sum _{i=1}^{n}( y_{i} -\hat{y_i})=-2\sum _{i=1}^{n}( y_{i} -\hat{m}x_{i} -\hat{c}) =0
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/dGJQE8hDa7.svg"></div>

Simplying the second equation (to make the process easier),

<!-- $$
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
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/XkEYTxG0Va.svg"></div>

We now need the value of <!-- $\hat{m}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/8wtsBengik.svg">, We'll solve this by using the value of <!-- $\hat{c}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/MXPDuZxw2v.svg"> we obtained above:

<!-- $$
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
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/aG58CbBlv2.svg"></div>

Uff, that took sometime with <!-- $\LaTeX$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/7GduZdsITj.svg">. Hopefully, I did everything without any mistakes. It looks like I did! Coming back to the equations, we derived the solutions to find <!-- $\hat{m}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/nYRVMCOWF1.svg"> and <!-- $\hat{c}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/EXlQoLZZRj.svg"> this values gives the minimum value for our **SSE**. This is the basic structure of the **Ordinary Least Squares (OLS)** and this type of regression is called **Ordinary Least Squares Regression** (the name is pretty self explanatory).

This works well for one _independent variable_ ($$x$$), but what if we have more? This is where we have to deal with vectors and matrices! Till now we've dealt with scalars, in the next section we'll derive the solutions to **SSE** using **OLS** for matrices. Buckle up for some matrix algebra!

### OLS for matrices

This kind of regression comes under **multiple regresion** where we have multiple input features with one real valued output. Our <!-- $x_i \in R^k$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/7q1EZmXMzx.svg"> i.e any point <!-- $x_i$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/7DLPBV0EhN.svg"> can be represented as <!-- $(x^1, x^2..,x^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/cHY1jeXS5z.svg">.

We need to change the notation here, from now on :

- <!-- $\mathbf{X}_{r\times c}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/WEM6uD1wGd.svg"> (bold uppercase letter) represents a matrix with $$r$$ rows and $$c$$ columns.
- <!-- $\mathbf{x}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/GVXMEdQCcs.svg"> (bold lowercase letter) represents a vector.
- <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/0Q01f5XtGo.svg"> (lowercase letter) represents a scalar.

I'll mention the dimension for the matrix where ever possible, cause this had been the cause for most of my _confusion_. I'll first walkthrought how we represent inputs in multiple regression.

<!-- $$
\mathbf{x}_i = [x^{1}_i, x^{2}_i.., x^{k}_i]\\
\small{\text{ (Please note that these superscripts are not powers, they are dimensions) }}\\
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/1Wbu2H35Yv.svg"></div>

Each input vector has $k$ features.

Now our matrix <!-- $\mathbf{X}_{n\times k}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/CktneOXDRN.svg"> contains <!-- $\mathbf{x}_i$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/VMt4xvL7VV.svg"> (<!-- $1 \times k$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/zJUyM0zlKi.svg"> dimension) as its rows

<!-- $$
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
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/w301HXQ1gN.svg"></div>

Now our equation becomes,

<!-- $$
\mathbf{\hat{y}} \text{ }(n \times 1 \text{ dimension}) = \mathbf{X}_{n \times k}\mathbf{W}_{k \times 1}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/AnOTlrEAUm.svg"></div>

This is an equation of a **hyperplane**. Now we need to introduce the _intercept_ term (Similar to <!-- $\hat{c}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/WBtBqNrLTl.svg"> in <!-- $\hat{y}=\hat{m}x+\hat{c}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/QLykiu1499.svg">). For this, I think it's better to start with 2D again.

We'll try to vectorize the 2D equation:

<!-- $$
\hat{y} = \left[ {\begin{array}{cc} x \text{ } 1 \end{array} } \right] \left[ {\begin{array}{cc}
    \hat{m} \\
    \hat{c} \\
  \end{array} } \right] = \hat{m}x + 1 \cdot \hat{c}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/MZ7APvhwoe.svg"></div>

Observe the <!-- $1$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/e0jU768Bq1.svg"> appended to the input vector. This acommodates for the intercept term. Using this intuition, we'll now add the intercept term to our equation. Our input matrix <!-- $\mathbf{X}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/zX7hTGanUW.svg"> becomes:

<!-- $$
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
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/XDXrYFfZ13.svg"></div>

Our _weight_ (or slope) vector <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/jK0fYvUH5F.svg"> becomes:

<!-- $$
\mathbf{W}_{(k+1) \times 1} = \left[ {\begin{array}{cc}
    w_1 \\
    w_2 \\
    . \\
    . \\
    w_k\\
    w_0
  \end{array} } \right]
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/S0JwYPIzhH.svg"></div>

For historic reasons (not really), we'll now refer to our interept term with <!-- $w_0$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/KmUyKHQ2FG.svg">. Finally rewriting our equation:

<!-- $$
\mathbf{\hat{y}} \text{ }(n \times 1 \text{ dimension}) = \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/rjvb7dJCZQ.svg"></div>

Ok, we've formulated our equation. Next comes the loss function, which will be the same **Sum of Squared Error (SSE)** we've seen earlier:

<!-- $$
\text{Error in prediction} = \mathbf{y} - \mathbf{\hat{y}} \\
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/oUZuW4cVAz.svg"></div>

We need to square this to avoid negative error:

<!-- $$
{\begin{split}
    \text{SSE} & = (\mathbf{y} - \mathbf{\hat{y}})^2 = (\mathbf{y} - \mathbf{\hat{y}})^T(\mathbf{y} - \mathbf{\hat{y}}) = (\mathbf{y} - \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1})^T(\mathbf{y} - \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1}) \text{ } & [\text{Using the result } \mathbf{X}^2 = \mathbf{X}^T\mathbf{X}] \\

    & \implies (\mathbf{y}^T - \mathbf{W}_{1 \times (k+1)}^T\mathbf{X}_{(k+1) \times n}^T)(\mathbf{y} - \mathbf{X}_{n \times (k+1)}\mathbf{W}_{(k+1) \times 1}) \text{ } & [\text{Using the result } (\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T] \\

    & \implies \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X}\mathbf{W} - \mathbf{W}^T\mathbf{X}^T\mathbf{y} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W} \text{ } & [\text{I won't mention dimensions anymore } ] \\
\end{split} }
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/hgz4ikQJE8.svg"></div>

Now, <!-- $\mathbf{y}^T\mathbf{X}\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/U3JqULyiDf.svg"> and <!-- $\mathbf{W}^T\mathbf{X}^T\mathbf{y}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/Yys17J1pgE.svg"> produces the same result (feel free to try it on few examples).

<!-- $$
\text{SSE} = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{W} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/nq15ebOtY2.svg"></div>

The next step is to take the derivative of SSE with respect to matrix <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/DhiwnqpNpn.svg">. Since we're dealing with set of variables (the constant and regression coefficients) we again use partial differentiation:

<!-- $$
\begin{split}
  \frac{\partial SSE}{\partial \mathbf{W}} & = \frac{\partial }{\partial \mathbf{W}} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{W} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}) \\
  & \implies \frac{\partial}{\partial \mathbf{W}}(\mathbf{y}^T\mathbf{y}) - \frac{\partial}{\partial \mathbf{W}}(2\mathbf{y}^T\mathbf{X}\mathbf{W}) + \frac{\partial}{\partial \mathbf{W}}(\mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}) \\
\end{split}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/sXfcze22OG.svg"></div>

This looks like something new if you don't know calculus on matrices. I'll try to map this with linear context.

The first term <!-- $\mathbf{y}^T\mathbf{y}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/W9prdBv58H.svg"> is a constant when differentiating with respect to <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/k2BSkhJhyO.svg">. So this will be <!-- $0$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/ZxlDGWrX3m.svg">.

The second term <!-- $2\mathbf{y}^T\mathbf{X}\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/SGBuGjkeGe.svg"> this is a "linear form" in <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/48emaCnwsu.svg"> and is equivalent to scalar term in which the variable we are differentiating with respect to is raised to the first power. This means that we can get the derivative by dropping the <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/jl9HSmMRXV.svg">. Since this is matrix calculus, we need to takt the transpose of whatever's remaining, giving us <!-- $-2\mathbf{X}^T\mathbf{y}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/Lu0X7kyi3z.svg">.

The final term <!-- $\mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/Buaw5bFuWu.svg">, is a "quadratic form" and is equivalent to scalar term in which the variable we are differentiating with respect to is raised to the second power. This indicates that we can get the derivative by dropping <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/8lMLG85tGl.svg"> and then taking the transpose, finally multiplying this by 2 giving us <!-- $2\mathbf{X}^T\mathbf{X}\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/BLZZEi2H4O.svg">.

<!-- $$
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
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/f8PqaKKpFi.svg"></div>

Now, that gives us the corresponding _weights_ that can be used to construct a hyperplane that best fits our data <!-- $\mathbf{X}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/ygxO8yp2xB.svg">. This equation is called **Normal Equation**. This is the naive mathematicians way of dealing with _linear regression_. In the next section, we'll do this in a different and common way through **gradient descent** (the ML guy way!).

### Gradient Descent

Gradient descent, it's everywhere in ML. Whenever you need to optimise a function in ML gradient descent (Or some of its variants) is the way. Understanding this (correctly) will make us look these classic ML algorithms through a different perspective (clearer perspective).

Let me start with what actually gradient descent is:

> "Gradient descent is a **first-order** iterative **optimization algorithm** for finding a local minimum of a **differentiable function**."

That's the definition I've found on [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20first-order%20iterative%20optimization%20algorithm,as%20gradient%20ascent..%20Gradient%20descent%20is%20generally%20). Surely it makes some sense, if not don't worry I'll take you through it.

- **optimization algorithm**: An optimization algorithm is a procedure which is executed iteratively by comparing various solutions till an optimum or a satisfactory solution is found. In gradient descent context, satisfactory result indicated global minima.
- **first-order**: In order to find the mentioned satisfactory results, we'll rely on first-order derivatives in first-order optimization algorithms.
- **differential function**: We need the optimization function to be derivable in order to find its derivative.

This algorithm basically finds a set of parameters that makes the value of the function minimum.

![GD hill image](https://tahaashtiani.com/assets/images/2020-01-01-Building-Gradient-Descent-for-Multivariate-Regression/img_7.png)

Now let's look at how gradient descent works. Assume that you roll a ball from the top of a hill (shown above). It eventually rolls to a pit (Or a deep hole). This pit represents our minima (May not be global minima).

_The gradient gives us the direction of steepest ascent. Remember we're dealing with vectors when we talk about gradient (like derivative for scalar). So, the sign of the gradient gives us the direction and the magnitude gives us the value of the ascent i.e very steep functions have high gradient, smooth and plain functions have low gradients. What we do in gradient descent is that we'll reverse the direction of the gradient so that instead of reaching the steepest ascent, we'll actually face the direction of the steepest descent! How genius is that?_

_Now we'll update our function such that the resultant is directed towards the minima. We add the negative of the gradient to our original function. We don't arbritraily add the gradient directly, instead we'll multiply it with something called **learning rate**_ <!-- $\eta \in (0.0001, 1)$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/pYlRrjXOwK.svg">_. This ensures that we don't skip the minima when the gradient is very high (function is very steep)._

Let's connect this to our SSE, this is the function we need to optimize.

<!-- $$
\text{SSE} = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{W} + \mathbf{W}^T\mathbf{X}^T\mathbf{X}\mathbf{W}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/pZljJxAJFM.svg"></div>

Let's say that we start off witha random <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/XLY1T6fQt0.svg"> matrix. We then use the below equation to update our <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/NoRkmDzSbg.svg"> matrix with gradient descent.

<!-- $$
\mathbf{W} = \mathbf{W} - \eta \nabla\text{SSE}
$$ -->

<div align="center"><img style="background: white; padding: 20px;" src="/svg/Crkt4N6KZN.svg"></div>

- When <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/tYcAFEyDMX.svg"> has the same sign as <!-- $\eta \nabla\text{SSE}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/MXuD0TE0ym.svg"> indicates that <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/dl2XYO37la.svg"> is towards the steepest ascent. We then perform a vector addition with <!-- $- \eta \nabla\text{SSE}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/7mmhbk6GWg.svg"> so that the magnitude of <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/r1myXAkuhh.svg"> gets decreased (we're taking <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/EH7ufcA5h0.svg"> towards steepest descent).
- When <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/iCgPOjDS1X.svg"> and <!-- $\eta \nabla\text{SSE}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/hQscWQyywa.svg"> have opposite signs indicates that <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/hYHQxRCwk5.svg"> is towards the steepest descent. We then perform a vector addition with <!-- $- \eta \nabla\text{SSE}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/6HGlc57JkE.svg"> so that the magnitude of <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/UuMHpktwaW.svg"> gets decreased further.

This happens till we get a <!-- $0$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/DgTmIkgKVB.svg"> gradient (remember that 0 indicates a global optimum, global minima in this case) where there would be no update to <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/1TuWnMGpdM.svg">. This will be (almost) the same <!-- $\mathbf{W}$ --> <img style="transform: translateY(0.1em); background: white;" src="/svg/3YA5uIM2H0.svg"> matrix we got earlier with our _normal equation_

I guess now you got how gradient descent works!

### Closing Notes

That's the mathematical reason behind the first ML algorithm you've probably learnt! Stay tuned for more :) Also, do let me know in the comments if you find any error or any suggestion.

I post all the remaining classic ML algorithms and few techniques on weekly basis. Do keep checking out. Try subscribing to my [RSS Feed](https://feedburner.google.com/fb/a/mailverify?uri=chandrakiran/VDNx){:target="\_blank"} to receive updates.
