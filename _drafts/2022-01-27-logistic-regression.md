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

[Linear Regression](https://chandu-4444.github.io/2021/12/07/linear-regression.html) can be used for classification problems. But it has some serious drawbacks. Let's look at it by considering an example.

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

Now, let's use linear regression to classify the following data points. Remember from [my previous post](https://chandu-4444.github.io/2021/12/07/linear-regression.html) that the objective of the linear regression is to mimimize the difference between the actual value and predicted value. Simply put, it finds a line that fits the data well. So, the bets-fit line for the data above will be as follows:

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

Look at that! The range of logistic function is between $0$ and $1$ (Our desired property for classification). Since the problem with our so called "linear regression classification" model is its range $[-\infty, \infty]$, why not wrap it with sigmoid function? That's the main idea of logistic regression. Let's formulate this mathematically:

$$
z = \hat{m}x + \hat{c} \\

f(z) = \frac{1}{1+e^{-z}} \\

$$

Now our final prediction $f(z)$ will have nice probablistic meaning, it gives the probability of z belonging to class $1$. Since there are only two class labels $0$ and $1$, this also means that the probability of $z$ belonging to class $0$ can be found using $1-f(z)$. 

Let's look at the figure if we use logistic regression for imbalanced data we've seen before.

<center>
   <figure>
        <img src="https://miro.medium.com/max/459/1*GgUYnIinlmTUnEc8CEesjQ.png
" style="border: 1px solid black; max-width: 70%;">
        <figcaption>Using Sigmoid function for imbalanced dataset</figcaption>
    </figure>
</center>

This is perfect! Now how do we find this fit? The only variables we aren't sure about in our sigmoid are $\hat{m}$ and $\hat{c}$ . If we were to find these parameters as we did for linear regression, we would be so wrong. The objective of the linear regression is to find the best fit line that minimizes the distance between predicted value and actual value. This is not the case for logistic regression. So, we'll now derive the loss function for logistic regression.

### Loss Function for Logistic Regression

The loss function for linear regression is **squared loss**. We found the optimal parameters by minimising this loss. What is the loss function for logistic regression? To answer this, we need to get through another question: **"When can we say that we have a good logistic model?"**. The answer to this is the higher the number of correctly classified points, the better is the model. 

If model **A** classifies $90$ out of $100$ points correctly and model **B** classifies $60$ out of $100$ points correctly, we say that model **A** is better than model **B** (**A** out performs model **B**). We need to maxmize this count. But only maximizing the count will not be sufficient! Since the outputs of the logistic function are probabilities, we need to penalize the misclassification proportional to the probability. 

To understand this, consider two points $x_1, x_2$ both belonging to class $1$. Our model predicted the probability of $x_1$ belonging to class $1$ as $0.47$ and $x_2$ belonging to class $1$ as 0.1 . Since both the points belong to class $1$, we need both the predicted values to be greater than $0.5$. We need to punish our model for this *misclassification*. But the punishment for misclassification for point $x_1$ has to be less than the punishment for point $x_2$. To achieve this, we need to consider the probabilities too in our loss function.

### Likelihood Estimation

Given all our $x_i$'s and for given parameters $\hat{m}$ and $\hat{c}$ , what is the distribution of $y_i$'s? If we were able to find this distribution, then we can sample that distribution to find the labels of any given data. Putting this statement mathematically we get $P(\mathbf{y}|\mathbf{X};\mathbf{W})$ called the **likelihood function** (function of $\mathbf{W}$) means that "Given $\mathbf{X}$ (the matrix which contains all $x_i$'s) and $\mathbf{W}$ (the matrix containing our parameters), what is the probability that the outcome of the distribution is $\mathbf{y}$?". Now let's stress upon the term **"distribution"**, this distribution if determined by $\mathbf{X}$, $\mathbf{W}$, and $\mathbf{y}$. But the only thing we aren't sure about is the matrix $\mathbf{W}$. It is obvious that we need to find this $\mathbf{W}$, but on what condition? Remember that for linear regression, we found this $\mathbf{W}$ by minimizing the squared loss. How do we do this in case of logistic regression? 

For a given $x_i$ we're predicting the probability of getting a $y_i$ as output paremeterized on $\mathbf{W}$. We need to maximize this probability so as to find the distribution that gives high probability for each $y_i$ (We're need to be highly certain that the label of the given input is $y_i$). That is, we need to maximize the likelihood function. This is called **maximum likelihood estimation**. Mathematically,

$$
L(\mathbf{w}) = P(\mathbf{y}|\mathbf{X};\mathbf{W})
$$

Since each obesrvation is independent of each, we can write it as (Recall that independent events in conditional probability can be expressed as product):

$$
L(\mathbf{W}) = \Pi_{i=1}^{n}P(y_i|x_i;\mathbf{W})
$$

We need to find the matrix $\mathbf{W}$ that maximizes this.

 But doing that would be painful (atleast in this form) since there are many  multiplications in our function. Instead of maximizing the function $L(\mathbf{W})$, we can maximize any strictly incerasing function of $L(\mathbf{W})$. What if we change all the multiplications to addition? Yeah, you're right! we'll use **log** to do this:

$$
\begin{split}
log(L(\mathbf{W})) & = \log(\Pi_{i=1}^{n}P(y_i|x_i;\mathbf{W})) \\
& = \sum_{i=1}^n\log(P(y_i|x_i;\mathbf{W}))\\
\end{split}
$$

Now, this seems good to differentiate! Note that **log** is strictly increasing function, so maximizing this will result in maximizing $L(\mathbf{W})$.

### Maximum Likelihood Estimation for Logistic Regression

Recall that the out of our $f(z)$ or $f(\hat{m}x+\hat{c})$ gives the probability of $x$ belonging to class 1. Using this, we can write:

$$
P(y=1|x;(\hat{m}, \hat{c})) = f(\hat{m}x+\hat{c}) \\
P(y=0|x;(\hat{m}, \hat{c})) = 1 - f(\hat{m}x+\hat{c})
$$

 The above expressions can be merged as:

$$
P(y|x;(\hat{m}, \hat{c})) = (f(\hat{m}x+\hat{c}))^y(1-f(\hat{m}x+\hat{c}))^{1-y}
$$

The above equation can be assumed as a likelihood function for $\hat{m}, \hat{c}$:

$$
\begin{split}
L(\mathbf{W}) & = P(\mathbf{y}|\mathbf{X;\mathbf{W}}) \\
& = \Pi_{i=1}^nP(y_i|x_i;(\hat{m}, \hat{c})) \\
& = \Pi_{i=1}^n(f(\hat{m}x_i+\hat{c}))^{y_i}(1-f(\hat{m}x_i+\hat{c}))^{1-y_i}
\end{split}
$$

We can apply **log** on both sides:

$$
\begin{split}
l(\mathbf{W}) & = \log(L(\mathbf{W})) \\
& = \sum_{i=1}^n y_i\log(f(\hat{m}x_i+\hat{c})) + (1-y_i)\log(1-f(\hat{m}x_i+\hat{c}))
\end{split}
$$

Now that's our objective function. We need to find $\hat{m}, \hat{c}$ that maximizes the above function. Hey wait! We knew about **gradient descent** that finds the minimum of a function. How do we find the maximum using gradient descent? Well, we can use the _negative_ of the above function and find the minimum using gradient descent. Our MLE now turned into a gradient descent optimization function.


