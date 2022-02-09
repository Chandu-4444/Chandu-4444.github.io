---
layout: post
title: "Logistic Regression: A deep dive"
subtitle: "Let's look at one of the most used classification technique in detail"
background: ""
tags: theory maths
use_math: true
comments: true
---
* toc
{:toc}

### What is classification?

Classification refers to the technique that segregrates the points into their respective classes. Classification is more extensively used in real world applications than compared to regression. Some application of classification are:

* Credit card fraud detection.
* Face recognition.
* Sentiment analysis.
* Speech recognition.
* Email spam detection.

Looking at the above applications will tell you how useful and important classification is!
There many algorithms to perform classification. We'll discuss the most popular **Logistic Regression** in this post. Logistic regression is a linear model. It assumes that the target classes are linearly seperable. It cannot be used for non-linear classification.

Let's get started with this. Please note that I'll always refer to binary classification (two-class classification) unless otherwise specified.

### Getting the intuition

Getting the intuition behind _logistic regression_ correctly is very rewarding trust me! MLP (Multi Layered Perceptron), the fundamental unit of Deep Learning can be viewed as logistic regression. This the reason why we not only solely look at the algorithm, but we'll derive it step by step. Buckle up for the ride.

<center>
    <figure>
        <img src="/img/posts/logistic-regression/hyperplane.jpg" style="border: 1px solid black; max-width: 70%;">
        <figcaption>Hyperplane $\pi$ that seperates both the classes</figcaption>
    </figure>
</center>

Look at the above figure. The line $\mathbf{\pi}$ seperates both the classes. In higher dimensions, the line becomes hyperplane.  
The goal of any classification algorithm is to segregrete the input points into their respective classes. Now, in this case we need to find such a hyperplane that classifies the _blue_ and _green_ points correctly ($$y_i \in \{-1, 1\}$$). Simply put, we need to _find the equation of line_ $\mathbf{\pi}$ that seperates them.

Let's consider the unit normal to the line 
$$\mathbf{w}$$ 
($$ \vert\vert \mathbf{w} \vert\vert = 1$$). Recall that the magnitude fof the expression $\mathbf{w}^T \mathbf{x}_p$ gives the distance of the point $\mathbf{x}_p$ from the line $\mathbf{\pi}$. Also, look and recall the result below: 

$$
\Bigg\{ 
    {\begin{array}{cc}
        \mathbf{w}^T \mathbf{x}_p > 0 & \text{ if }\mathbf{x_p} \text{and the } \mathbf{w} \text{ lies on the same side of the line}\\
        \mathbf{W}^T \mathbf{x}_p < 0 & \text{otherwise}
    \end{array}} 
\Bigg\}
$$

Just make sure to recall the above concept. We'll now make our first attempt to derive a mathematical formulation for classification. Assume that for _green_ points $y_i = +1$ and for _blue_ points $y_i = -1$. For _green_ points, $$\mathbf{w}^T \mathbf{x}_i > 0$$ and $$y_i = +1$$ and for _blue_ points $$\mathbf{w}^T \mathbf{x}_i < 0$$ and $$y_i = -1$$. Using these two facts, we can frame the below hypothesis:

$$
\Bigg\{ 
    {\begin{array}{cc}
        y_i \mathbf{w}^T \mathbf{x}_p > 0 & \text{Point } \mathbf{x}_p \text{ was classified correctly}\\
        y_i \mathbf{W}^T \mathbf{x}_p < 0 & \text{Point } \mathbf{x}_p \text{ wasn't classified correctly}\\
    \end{array}} 
\Bigg\}
$$

That looks promising and makes sense. The only question remains is how to find $$\mathbf{w}$$? We need to construct an **objective function** that measures how well we have classified our data points. From our derivation above, we can formulate our objective function as follows:

$$
\mathbf{w}^* = \arg \max_{\mathbf{w}} \sum_{i=1}^n y_i \mathbf{w}^T \mathbf{x}_i\\
\mathbf{w}^*\text{ is the optimal solution to the above equation}
$$

Since $$y_i \text{ and } \mathbf{x}_i$$ are already known and fixed, our objective is to find such optimal $$\mathbf{w}^*$$ that maximizes the above expression. This ensures that most of the terms from $$i=1$$ to $$i=n$$ in the objective function has to be positive i.e classified correctly. Are we done? No, the objective function has some drawbacks. Let's see what are those and how to overcome them.

### Drawbacks Of Our Initial Objective Function

<center>
    <figure>
        <img src="/img/posts/logistic-regression/objective_function.jpg" style="border: 1px solid black; max-width: 70%;">
        <figcaption>An ideal hyperplane $\mathbf{\pi}_1$ that seperates both the classes. But the objective function isn't maximized.</figcaption>
    </figure>
</center>

Look at the above figure. The line $\pi_1$ does a good job in seperating _orange_ and _blue_ points. It miss classified only one _blue_ point as _orange_. Let's now calculate the value of our objective function for this line:

$$

\text{Value of objective function for } \pi_1 = \sum_{i=1}^{11} y_i \mathbf{w}^T \mathbf{x}_i \\

    {\begin{array}{cc}
          = +1(+1)+1(+1)+1(+1)+1(+1)+1(+1) + [-1(-1)]+ [-1(-1)]+ [-1(-1)]+ [-1(-1)]+ [-1(-1)] + [-1(+100)] \\
          = 5 + 5 - 100 = 90
    \end{array}} \\
$$

We can clearly see that even though the line seems to be a good one, the objective function is not maximized. This is because of one _blue_ point that has been classified incorrectly. The penalty for this misclassification is -100. This effected our objective function a lot. This is an **outlier**. Outliers are the abnormal data points that occurs in the dataset. These outliers must be handled with care. This outlier effected our objective function a lot. But we cannot deny the fact that the line is a good one. the only drawback is that the measure we used to evaluate it is not correct. We need to modify our objective function.

### Formulating the correct objective function

The idea is to **squash** the signed distance ($$\mathbf{w}^T\mathbf{x}_i$$) to a value between 0 and 1. If the value of the signed distance is positive, we squash it to 1 (We take class $$1$$ to be on the same direction as $$\mathbf{w}$$). If the value of the signed distance is negative, we squash it to 0 (We now assume our class label to be $$0$$). We do this squashing using **sigmoid function**.

<center>
    <figure>
        <img src="/img/posts/logistic-regression/sigmoid.png" style="border: 1px solid black; max-width: 70%;">
        <figcaption>Sigmoid Function</figcaption>
    </figure>
</center>

Sigmoid function has a range between $$[0, 1]$$. This makes it perfectly suitable for binary classification tasks. Looking at the below equation can tell you a lot about the sigmoid function.

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$