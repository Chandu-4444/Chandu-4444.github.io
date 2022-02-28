---
layout: post
title: "Logistic Regression"
subtitle: "Let's look at one of the most used classification technique in detail"
background: ""
tags: theory maths
use_math: true
comments: true

---

* toc
  {:toc}

### What is Classification?

Classification refers to the technique that segregrates the points into their respective classes. Classification is more practically used in real world applications. Some application of classification are:

* Credit card fraud detection.
* Face recognition.
* Sentiment analysis.
* Speech recognition.
* Email spam detection.

Looking at the above applications will tell you how useful and important classification is!
There many algorithms to perform classification. We'll discuss the most popular **Logistic Regression** in this post. Logistic regression is a linear model. It assumes that the target classes are linearly seperable. It cannot be used for non-linear classification.

_Note that I'll always refer to binary classification (two-class classification) unless otherwise specified._

### Why not use Linear Regression instead?

[Linear Regression](https://chandrakiran.tech/2021/12/07/linear-regression.html) can be used for classification problems. But it has some serious drawbacks. Let's look at it by considering an example.

Let's say we have a [balanced dataset](#balanced-vs-imbalanced) of $20$ people. Our dataset contains the information about customers whether they made a purchase or not based on their age. Here, **"purchased"** is a binary variable $0$ (they didn't purchase) or $1$ (they made a purchase). 

<div class="todo-container" id="balanced-vs-imbalanced">
<span class="todo-icon">💡</span> <span class="todo-head">What is a balanced dataset?</span>

<div class="todo-body">
There are two types of data for classification problems: 
<p>1. <b>Balanced data</b>: The labels are equally distributed among all the classes. Each class contains same number of data points. </p>
<p>2. <b>Imbalanced data</b>: The labels are not equally distributed. One class may contain many points, while the other contains very few points. </p>
</div>
</div>

The plot our dataset looks like this:

<center>
   <figure>
        <img src="https://miro.medium.com/max/459/1*n6wG6BdkXJegoFo9G7rg8g.png" style="border: 1px solid black; max-width: 60%;">
        <figcaption>Dataset Plot</figcaption>
    </figure>
</center>

Now, let's use linear regression to classify the following data points. Remember from [my previous post](https://chandrakiran.tech/2021/12/07/linear-regression.html) that the objective of the linear regression is to mimimize the difference between the actual value and predicted value. Simply put, it finds a line that fits the data well. So, the bets-fit line for the data above will be as follows:

<center>
   <figure>
        <img src="https://miro.medium.com/max/466/1*lBc9kCcWfZxlvX-gdrVoUA.png" style="border: 1px solid black; max-width: 60%;">
        <figcaption>Best-fit line for the data</figcaption>
    </figure>
</center>

The **red** line is the best fit line for the data given. The predicted $y$ values ($\hat{y}$) are as follows:

```markdown
+-----+-------------------+------+
| Age | Predicted Y Value | label|
+-----+-------------------+------+
|  10 |        1.21428571 |   1  |
|  15 |        0.83834586 |   1  | 
|  19 |        0.53759398 |   1  |
|  20 |        0.46240602 |   0  |
|  25 |        0.08646617 |   0  |
|  30 |       -0.28947368 |   0  |
+-----+-------------------+------+
```

The prediction of the class labels is simple. If the $\hat{y} >0.5$, predict the label as $1$ else $0$. This seems like almost **"the perfect way"** for classification. But as I mentioned before, there are two drawbacks for this.

**1. The predicted $\hat{y}$ is continous, not probabilistic.**

The predicted $y$ values ($\hat{y}$) have no significance other than the fact that they are compared to $0.5$ to predict the class labels. We don't care if the $\hat{y}$ for $10$ is $1.2$, we only care that it is greater than $0.5$. We ideally want a meaning to the predicted $y$ values. What if the values depict the probability? That would be great! We can directly label the data points by looking at the predicted values.  The probability of any event lies between $0$ and $1$. Unfortunately, our linear regression can range anywhere between $[-\infty, \infty]$. So we cannot simply use linear regression for classification.

**2.  Sensitive to Imbalanced Data**

This method isn't suitable for imbalanced data. Let's add few more data points from age $60-70$ that belongs to class $0$. The best fit line will now be as follows:

<center>
   <figure>
        <img src="https://miro.medium.com/max/466/1*HFrHo41Fe9Zqo32v4_Sbcw.png" style="border: 1px solid black; max-width: 60%;">
        <figcaption>Best-fit line for the data</figcaption>
    </figure>
</center>

Since the objective of the linear regression is to minimize the squared loss, it moved a bit towards bottom (i.e sensitive to the change in data). If you observe carefully, you can notice that the customers between age $20-23$ are wrongly predicted

```markdown
+-----+-------------------+------+
| Age | Predicted Y Value | label|
+-----+-------------------+------+
|  20 |        0.53687781 |   1  |
|  21 |        0.52284026 |   1  |
|  22 |        0.50880271 |   1  |
|  23 |        0.49476516 |   1  |
+-----+-------------------+------+
```

This imbalance affected our performance. 

So, it is very clear that we cannot use linear regression for classification (not directly ofcourse!). We need to modify this to work for classification. We do that with the help of **Logistic  Function**.

### The Logistic Function

According to [Wikipedia]([Logistic regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)):

> The **logistic model** (or **logit model**) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.

So, a **logistic model** is used to predict the probability of a point belonging to a certain class ($0 \text{ or } 1$ in our case). In **Logistic Regression**, we do this using a modified form of a function called **Logistic Function**. The function is represented as follows:

$$
f(x) = \frac{L}{1+e^{-\frac{1}{s}(x-x_0)}} \\
\begin{split}
& x_0 \text{: The x value of sigmoid's midpoint.} \\
& \text{L: The curve's maximum value.} \\
& \text{s: The logistic growth rate or steepness of the curve.} \\
\end{split}
$$

For classification (binary or two-class) problems, we often have $0$ and $1$ as our class labels (So, $L=1$). From this, we can deduce that our midpoint of the function should occur at  $x=0$ i.e $f(x)=0.5$. These values put into logistic function is called **The Standard Logistic Function** (or **Sigmoid Function**).

$$
f(x) = \frac{1}{1+e^{-x}} \\
\begin{split}
& x_0 = 0.5 \\
& L = 1 \\
& k = 1 \\
\end{split}
$$

The diagram of the **Sigmoid Function** is as follows:

<center>
   <figure>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png" style="border: 1px solid black; max-width: 70%;">
        <figcaption>Sigmoid Function</figcaption>
    </figure>
</center>

Look at that! The range of logistic function is between $0$ and $1$ (Our desired property for classification).
