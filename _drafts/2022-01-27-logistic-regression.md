---
layout: post
title: "Logistic Regression: A deep dive"
subtitle: "Let's look at one of the most used classification technique in detail."
background: ""
tags: theory maths
use_math: true
comments: true
---
* toc
{:toc}

### What is classification? 

If you've gone through [my previous post about linear regression](https://chandrakiran.tech/2021/12/07/linear-regression.html), then there is high chance that you know what regression is. Regression is generally refers to mapping the input data to a continuous target variable i.e $$y \in R$$. In classification, the target variable is a discrete set of values i.e $$y$$ is countable. Classification is more extensively used in real world applications than compared to regression. Some application of classification are:

* Credit card fraud detection.
* Face recognition.
* Sentiment analysis.
* Speech recognition.
* Email spam detection.

Looking at the above applications will tell you how useful and important classification is! Let's start learning it.

### Why new algorithm for this? Can't we just use linear regression for this?

That is the first genuine question one can ask. Since classification algorithm can be thought of as a special case of regression, we can use linear regression for classification assuming our target variable is a discrete variable. But the problem is that linear regression is linear in nature and thus we can't use it for classification of non linear natured tasks. Let me show you what is a linear natured task and non linear natured task.



