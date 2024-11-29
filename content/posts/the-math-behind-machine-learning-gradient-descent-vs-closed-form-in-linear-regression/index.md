+++
title = "The Math Behind Machine Learning - Gradient Descent vs. Closed Form in Linear Regression"
date = "2024-11-29"
draft = true
pinned = false
+++
{{<lead>}}

Linear regression is a foundational concept in machine learning often seen as the first step into the world of predictive modeling. At first glance it seems simple - just a line through some points - but understanding the math behind it allows for a better grasp on optimization and machine learning principles. In this post we will explore the mathematical foundation of linear regression, compare the exact closed-form solution with the iterative gradient descent method and discuss why gradient descent  dominates in more complex machine learning models. If you're new to machine learning or just brushing up on the basics, this post will deepen your understanding of linear regression and its optimization techniques.

{{<lead>}}

# The Foundation of Predictive Modeling

Linear regression is one of the most fundamental concepts in machine learning (ML) and is used to predict house prices, analyzing trends and finding correlations. It's a simple but powerful tool for understanding relationships between data and is often the first model introduced to aspiring data scientists because it contains many core principles of optimization and prediction.

In this post we will look at the math behind linear regression in detail and find out how to minimize the error of a line fitted to data points. We will cover two approaches: 

1. The **closed-form solution** that gives us the exact answer using algebra
2. **Gradient descent** which is an iterative method commonly used in ML

We will also discuss why gradient descent is used in more advanced models and scenarios where the closed-form solution is impractical or even impossible.

By the end you will not only understand how linear regression works but also get a broader understanding of optimization techniques in ML and even learn how to implement both techniques in Python. Let's get started!

# Linear Regression and The Math Behind it

The goal of linear regression is to find the best-fit line *y = mx + b* that minimizes the error between the predicted value *mx + b* and the actual value *y*<sub>*i*</sub>. To quantify this error we use a so called loss function, in this case the **Mean Square Error** (MSE).