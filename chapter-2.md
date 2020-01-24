# Chapter 1

## Exercises

### Alternate backpropagation equations

Should be obvious by multiplying out the matrix multiplication.

### Proof of backpropagation equations 3 and 4

Backpropagation equation 4 comes from the same process as equation 2, but looking at the previous layer.

Equation 3 comes from applying the derivative of the z error with respect to the bias term.

### Backpropagation with a single modified neuron

*Suppose we modify a single neuron in a feedforward network so that the output from the neuron is given by $f(\sum_j w_j x_j + b)$, where $f$ is some function other than the sigmoid. How should we modify the backpropagation algorithm in this case?*

The backpropagation algorithm will change in the calculation for that specific neuron. Instead of using the derivative of the sigmoid function, we should use the derivative of the function $f$ that is being used.

### Backpropagation with linear neurons

*Suppose we replace the usual non-linear σ function with $\sigma (z) = z$ throughout the network. Rewrite the backpropagation algorithm for this case.*

In this case, the activations are all equal to the weighted inputs $z$. This means that the feedforward process can be computed identically other than for the activations. For the output error, we can see that it would equal the cost derivative at the output layer. Finally, for the backpropagation algorithm, we would take the linear derivative of 1 for the derivative of $\sigma(z)$.
