# TensorFlow Tutorials - Linear Regression

In this tutorial, we will be looking at how we can use TensorFlow to implement a linear regression model on a given data set. In our case, the data set is going to be very, very small compared to real life data sets that we will be looking at later in the series (our data set only has 4 points). Most of us just get hit by the term linear regression (if you have not taken some advanced math classes), linear regression, in layman terms is all about drawing the best fit line that best describes a given data set. So now that we know what linear regression means, let's get started.

First make sure you have TensorFlow installed on your computer and import it as follows. 

```python
import tensorflow as tf
```

Our ultimate goal as stated earlier, is to derieve a best fit line that best describes the given data set. A line as we all know follows the standard equation as shown below.

```math
y = mx + c
```

