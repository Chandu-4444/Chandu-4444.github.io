---
layout: post
title: "Getting Started with Machine Learning."
subtitle: "World isn't cupcakes & rainbows. I'll try my best to provide you a rough roadmap and resources to get started with Machine Learning."
background: ""
tags: resources books tips
use_math: true
comments: true
toc: true
---

### So, you're interested in learning ML?
> "A good start is half the work"

Yeah, I'm glad that you took a step to make a better version of yourself in ML. In this short article, I'll help you in _getting started with ML_. Me being a learner, and am still learning will try my best to point out resources, tips and few practices that I followed during my early stages of learning.

>"Getting started is always the hardest step"

That says it all! And trust me, this is true. But it's worth it and will truely boost your confidence. So let's get started!


### Mathematics
This is almost the first question that you've probably search for. I'm not going to tell you mathematics isn't important nor I'll tell you it's everything. You can build a great recommender system like Netflix did, or create that addictive algorithm like TikTok without any knowledge of mathematics. But the problem comes when you can't understand why they're so beautiful, why excatly they're so popular. These questions leads to the mathematics behind them. You'll probably never appreciate any ML algorithm until you get a grasp of mathematics behind them.

I remember a saying **"Machine Learning is for mathematicians who can program"**. This almost true to believe, only is it was the other way around **"Machine Learning is for programmers who knew mathematics"**. Yeah, that's prefect.

_Statistics and Probability_, _Calculus_, _Linear  Algebra_

These three were the pillars behind any machine learning algorithm. So if you want to get the essence of an algorithm in ML you should be comfortable with the above three, not equally ofcourse.

![Mathematics](/img/posts/getting-started/scatter-plot.svg)

Yes, to my experience you have to be **60%** familiar with P&S, **20%** familiar with Linear Algebra, **20%** familiar with Calculus (A rough estimation). By familiarity I mean here that if you know 6 chapters in P&S, you have to study 2 chapters in each Linear Algebra and Calculus. 

Now comes the resources part. Hmm let me tell you that I didn't bother much about mathematics during my early stages. But I definitely felt a gap. Unfortunately I didn't study mathematics seriously during my freshmen and sophomore year (I regret this even now). Luckily, the development comes when we realize our mistakes. I realized and started learning those mathematical concepts. I'll now list those:

- [Essential Mathematics for Machine Learning](https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE){:target="_blank"}

  - This is the first mathematics based _playlist_ I've watched. Let me be honest with you, I never completed this in one go. But I used it as a reference whenever I need help with mathematics. Few videos in this playlist takes sometime to make you feel comfortable (Ofcourse it depends).
- [Mathematics for Machine Learning](https://mml-book.github.io/){:target="_blank"}
  - This _book_ is a good one (if you know what to read). Yes! It is not necessary to read whole book, the first part is enough. Each chapter (in the first part) introduces you different mathematical concepts. Read every chapter until you feel comfortable. Keep in mind that few concepts need additional resources to understand (But if you feel it's way to deep, feel free to skip those concepts as I did :) ). 
  
This is a personal recommendation and I'm happy to address few more resources if needed in the comments.


### Python VS R
I personally prefer Python, owing to its user friendly syntax and flexibility. Honestly, programming isn't a big deal in ML. Yes! We are in thos days where we can implement a State-Of-The-Art ML algorithm in less than 20-30 lines of code. But there will be a handful amount of code that has to be typed for data preprocessing and visualization. You have to be familiar with Object Oriented Programming to feel comfortable while coding ML algorithms. You need to make sure that you're good with Jupyter Notebooks. You have to get accustomed with the following libraries:
- Numpy
- Matplotlib
- Scikit Learn
- Seaborn
- Pandas

I suggest the following book for this: [Python Data Science Handbook](https://www.amazon.in/Python-Data-Science-Handbook-Essential/dp/9352134915/ref=sr_1_1?hvadid=72980161955818&hvbmt=be&hvdev=c&hvqmt=e&keywords=python+data+science+handbook&qid=1641362580&sr=8-1){:target="_blank"}. The first four chapters will be more than enough.

### Courses
These are the courses that I took to get my self familiar with ML algorithms. The thing is, I took most of the courses before I'm confident with mathematics. I didn't worried about mathematical details as I initially didn't cared. After I learned enough mathematics, I was able to connect with the concepts I've learnt.

- [Machine Learning by Stanford](https://www.coursera.org/learn/machine-learning#syllabus){:target="_blank"}
  - You probably have heard about this course. This course will introduce with all the concepts in ML (without much mathematical detail). You can take this course and then connect the mathematical dots afterwards. This course has assignments in MATLAB but a simple google search will fetch you a repository that implements each assignment in Python.
- [Machine Learning Course by Coding Blocks](https://online.codingblocks.com/courses/machine-learning-course-online){:target="_blank"}
  - This is an interesting course that not only deals with a bit of maths but teaches you various implementations from scratch. This course has some interesting projects too.
  
### Books
I love reading books. The reason is that you can get to know much more thorought reading books. The following are the books I read during my early days of learing:
- [Python Data Science Handbook](https://www.amazon.in/Python-Data-Science-Handbook-Essential/dp/9352134915/ref=sr_1_1?hvadid=72980161955818&hvbmt=be&hvdev=c&hvqmt=e&keywords=python+data+science+handbook&qid=1641362580&sr=8-1){:target="_blank"}
- [Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow](https://www.amazon.in/Hands-Machine-Learning-Scikit-Learn-Tensor/dp/9352139054){:target="_blank"}
  
### Practice
_"Practice makes everyone perfect"_. Truely said! Though you've mastered mathematics, though you are good with all those ML algorithms you are nothing more than a beginner without practice. The following websites can be used for practice:
- [Kaggle](https://www.kaggle.com/){:target="_blank"}
  - I bet you know this. Kaggle is like a heaven. You can almost get anything you want on Kaggle (Datasets, tutorials, resources etc).
- GitHub
  - I know you didn't expect this. But this one is a great tip. You can get the things you didn't get on Kaggle here.
  
### My personal tips
I suggest going through the following order 
- [Machine Learning by Stanford](https://www.coursera.org/learn/machine-learning#syllabus){:target="_blank"}
- [Essential Mathematics for Machine Learning](https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE){:target="_blank"}
  - Read [Mathematics for Machine Learning](https://mml-book.github.io/){:target="_blank"} along with this if needed 
- Read few chapters from [Python Data Science Handbook](https://www.amazon.in/Python-Data-Science-Handbook-Essential/dp/9352134915/ref=sr_1_1?hvadid=72980161955818&hvbmt=be&hvdev=c&hvqmt=e&keywords=python+data+science+handbook&qid=1641362580&sr=8-1){:target="_blank"} and [Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow](https://www.amazon.in/Hands-Machine-Learning-Scikit-Learn-Tensor/dp/9352139054){:target="_blank"}
- [Machine Learning Course by Coding Blocks](https://online.codingblocks.com/courses/machine-learning-course-online){:target="_blank"}
- Finaly keep practising from [Kaggle](https://www.kaggle.com/){:target="_blank"}.

Apart from the above resources I'll keep sharing additional resources through my newsletter, so make sure you subscribe to it.

{% include disqus.html %}
