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

The goal of linear regression is to find the best-fit line *y = mx + b* that minimizes the error between the predicted value *mx + b* and the actual value *y*<sub>*i*</sub>. To quantify this error we use a so-called loss function, in this case the **Mean Square Error** (MSE). 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>L</mi>
  <mo>(</mo>
  <mi>m</mi>
  <mo>,</mo>
  <mi>b</mi>
  <mo>)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <mo>&#x2211;</mo>
  <msub>
    <mi>i</mi>
    <mn>1</mn>
  </msub>
  <mo>^</mo>
  <mi>n</mi>
  <mo stretchy="false">(</mo>
  <mi>y</mi>
  <msub>
    <mi>i</mi>
    <mrow/>
  </msub>
  <mo>-</mo>
  <mo stretchy="false">(</mo>
  <mi>m</mi>
  <mo>&#x22C5;</mo>
  <mi>x</mi>
  <msub>
    <mi>i</mi>
    <mrow/>
  </msub>
  <mo>+</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">)</mo>
  <msup>
    <mo stretchy="false">(</mo>
    <mn>2</mn>
  </msup>
</math>

Let's look at a couple of illustrations to visualize these concepts.\
\
Given some data points, here in blue, we're trying to find this red line that fits best to all points.



In this image we can see that we can adjust two parameters to make this line fit our data: the slope *m* and the y-intercept *b*.



In this illustration we can visualize the meaning of the MSE. In purple we can see the predtiction *mx<sub>i</sub> + b* at a point *x*<sub>*i*</sub>. The true value at that point is given by the blue points' y-value *y<sub>i</sub>*. The difference of the true value and the prediction gives us the error for one point (*y<sub>i</sub> - (mx<sub>i</sub> + b)*). To get the error for our entire dataset we square that value to get rid of any negative values, so the positive and negative values don't cancel out, and then sum the error of every point. At the end we normalize the error by dividing by the length of the dataset *n*.

Now that we know how the loss is calculated, let's explore two ways to minimize the error: The closed-form solution and gradient descent.