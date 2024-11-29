+++
title = "The Math Behind Machine Learning - Gradient Descent vs. Closed Form in Linear Regression"
date = "2024-11-29"
draft = false
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

## Loss functions

The goal of linear regression is to find the best-fit line *y = mx + b* that minimizes the error between the predicted value *mx + b* and the actual value *y*<sub>*i*</sub>. To quantify this error we use a so-called loss function, in this case the **Mean Square Error** (MSE). 

<p> L(m, b) = (1/n) &Sigma;<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - (mx<sub>i</sub> + b))<sup>2</sup> </p>

Let's look at a couple of illustrations to visualize these concepts.\
\
Given some data points, here in blue, we're trying to find this red line that fits best to all points.



In this image we can see that we can adjust two parameters to make this line fit our data: the slope *m* and the y-intercept *b*.



In this illustration we can visualize the meaning of the MSE. In purple we can see the predtiction *mx<sub>i</sub> + b* at a point *x*<sub>*i*</sub>. The true value at that point is given by the blue points' y-value *y<sub>i</sub>*. The difference of the true value and the prediction gives us the error for one point (*y<sub>i</sub> - (mx<sub>i</sub> + b)*). To get the error for our entire dataset we square that value to get rid of any negative values, so the positive and negative values don't cancel out, and then sum the error of every point. At the end we normalize the error by dividing by the length of the dataset *n*.

Now that we know how the loss is calculated, let's explore two ways to minimize the error: The closed-form solution and gradient descent.

## Optimization

### Gradient descent

Gradient descent is an optimization technique that starts with some initial guess (or random values) for its parameters, in this case m and b, and adjusts them iteratively. This iterative approach is the reason why gradient descent is used in more complex machine learning model, where finding an analytical solution is often impossible.

#### How does it work?

Gradient descent works by finding the "direction of steepest descent" on the loss function. At each step, we calculate the gradient of the loss function, which tells us how the error changes with respect to the parameters m and b. This gradient points in the direction where the loss increases the fastest. To minimize the loss, we move in the opposite direction of the gradient.

Imagine the loss function as a bowl-shaped surface, where the height represents the error. The gradient is like a compass pointing uphill, and the "steepest descent" is the opposite direction—downhill. By taking small steps in this direction, guided by the learning rate *α*, we iteratively find the minimum of the loss function.

The learning rate shouldn't be too small, because with very tiny steps, finding the way "down-hill" takes a long time. On the other hand the learning rate also shouldn't be too big, because by taking large steps we risk overshooting the minimum and going "up-hill" on the other side of the minimum.

#### How do you find the gradient?

To find the gradient we simply take the partial derivative of the loss function with respect to each variable. In this case we only have to find two partial derivatives: One with respect to *m* and one with respect to *b*. The partial derivatives measure how sensitive the loss is to changes in each parameter and can be calculated by using the chain rule and the power rule from calculus as shown in the calculations below. 

Now that we calculated the partial derivatives we can use them to iteratively adjust our parameters using the following update rules:



### Closed-Form Solution

Finding the closed-form solution is an analytical approach alternative to gradient descent to optimize the loss function. It offers an exact solution for the parameters *a* and *b* in one step. For a simple model, like linear regression, and a small dataset it's computationally efficient, but as complexity and size of the dataset grow, finding the closed-form solution becomed inefficient or even impossible.

#### How does it work?

In optimization, partial derivatives measure how a function changes with respect to each parameter. To find the minimum of the loss function, we need to identify where the function stops decreasing—this happens when the slope is zero. For the Mean Squared Error (MSE), this means setting the partial derivatives with respect to m and b to zero:



Imagine the loss function as a 3D bowl-shaped surface where the height represents the error. The minimum is at the bottom of the bowl, where the slope is zero in every direction. Mathematically, this is where the partial derivatives of the loss function are zero.

#### How do you find the optimal parameters?

To find *m* and *b* we can set their partial derivatives equal to zero and we get a 2x2 system of equations. If we solve this system of equations we find the optimal values for both parameters. The formulas for *m* and *b* are:



Here is how to solve this 2x2 system of equations to get to these formulas: