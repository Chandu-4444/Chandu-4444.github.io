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

Look at the above figure. The line $\pi$ seperates both the classes. In higher dimensions, the line becomes hyperplane.  
The goal of any classification algorithm is to segregrete the input points into their respective classes. Now, in this case we need to find such a hyperplane that classifies the _blue_ and _green_ points correctly ($$y_i \in \{-1, 1\}$$). Simply put, we need to _find the equation of line_ $\pi$ that seperates them.

Let's consider the unit normal to the line 
$$\mathbf{w}$$ 
($$ \vert\vert \mathbf{w} \vert\vert = 1$$). Recall that the expression $\mathbf{w}^T \mathbf{x}_p$ gives the distance of the point $\mathbf{x}_p$ from the line $\pi$. Also, look and recall the result below: 

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

Since $$y_i \text{ and } \mathbf{x}_i$$ are already known and fixed, our objective is to find such optimal $$\mathbf{w}^*$$ that maximizes the above expression. This ensures that most of the terms from $$i=1$$ to $$i=n$$ in the objective function has to be positive i.e classified correctly. Woo.. are we done? No. the objective function has some drawbacks. Let's see what are those and how to overcome them.

### Drawbacks Of Our Initial Objective Function





