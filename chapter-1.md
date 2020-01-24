# Chapter 1

## Exercises

### Sigmoid neurons simulating perceptrons, part I

*Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, $c > 0$. Show that the behaviour of the network doesn't change.*

The output of a perceptron is defined as:

$$\textrm{output} = \begin{cases}
0 \quad \textrm{if } w \cdot x + b \leq 0 \\
1 \quad \textrm{if } w \cdot x + b > 0
\end{cases}$$

In fact, the output depends on the following function $f$:

$$f(x) = w \cdot x + b$$

If we multiply all the weights and biases by a positive constant $c$ and re-examine the function $f$:

$$f(x) = c \cdot w \cdot x + c \cdot b$$
$$g(x) = \frac{1}{c} \cdot f(x) = w \cdot x + b$$

As long as $c$ is a positive constant, we will have the exact same behavior for the perceptron, since the result of the function $g$ will have the same sign as the original function $f$. In contrast, if $c$ is a negative number, then the perceptron's outputs will always be flipped.

### Sigmoid neurons simulating perceptrons, part II

*Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to the network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been fixed. Suppose the weights and biases are such that $w \cdot x + b \neq 0$ for the input $x$ to any particular perceptron in the network. Now replace all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a positive constant $c > 0$. Show that in the limit as $c \rightarrow \infty$ the behaviour of this network of sigmoid neurons is exactly the same as the network of perceptrons. How can this fail when $w \cdot x + b = 0$ for one of the perceptrons?*

We know that for any given perceptron, the following holds:

$$w \cdot x + b \neq 0$$

If we replace the perceptrons by sigmoid neurons, we have that each output is

$$f(x) = \frac{1}{1 + \exp(- \sum_j w_j x_j - b})$$

Since we know that the exponent on $e$ is not equal to $0$ for any neuron, we also know that the output for any of these sigmoid neurons is not equal to $\frac{1}{2}$ (since the exponent term will resolve to either a very small, positive number - yielding output close to 1 - or a large, positive number - yielding output close to 0). Multiplying the entire network's weights and biases by a large, positive constant $c$ will give us the output function

$$f_c(x) = \frac{1}{1 + \exp(- \sum_j c w_j x_j - c b)}$$
$$f_c(x) = \frac{1}{1 + \exp(-c (\sum_j w_j x_j + b))}$$

The exponent term will thus be amplified in the direction of the sign of the original function value. For $c \rightarrow \infty$, we will thus have

$$f_c(x) = \begin{cases}
0 \quad \textrm{if } \sum_j w_j x_j + b < 0 \\
1 \quad \textrm{if } \sum_j w_j x_j + b > 0
\end{cases}$$

which is very close to the definition of a perceptron. This breaks down if the result of $w \cdot x + b$ is $0$, since then we have exactly a "tie" - the output function would be

$$f_c(x) = \frac{1}{1 + \exp{(0)^c}}$$

for any $c$, which is

$$f_c(x) = \frac{1}{1 + \exp{0}}$$
$$f_c(x) = \frac{1}{1 + 1} = \frac{1}{2}$$

This gives us a result of exactly $0.5$ for values for which $w \cdot x + b = 0$. However, this gives us undefined behavior in the perceptron case, since by the definition of a perceptron, this situation would give us an output of $0$.

### Weights and biases for bitwise representation

*There is a way of determining the bitwise representation of a digit by adding an extra layer to the three-layer network above. The extra layer converts the output from the previous layer into a binary representation, as illustrated in the figure below. Find a set of weights and biases for the new output layer. Assume that the first $3$ layers of neurons are such that the correct output in the third layer (i.e., the old output layer) has activation at least $0.99$, and incorrect outputs have activation less than $0.01$.*

Weights:

0 -> a: -1
0 -> b: -1
0 -> c: -1
0 -> d: -1

1 -> a: 1
1 -> b: -1
1 -> c: -1
1 -> d: -1

2 -> a: -1
2 -> b: 1
2 -> c: -1
2 -> d: -1

3 -> a: 1
3 -> b: 1
3 -> c: -1
3 -> d: -1

4 -> a: -1
4 -> b: -1
4 -> c: 1
4 -> d: -1

5 -> a: 1
5 -> b: -1
5 -> c: 1
5 -> d: -1

6 -> a: -1
6 -> b: 1
6 -> c: 1
6 -> d: -1

7 -> a: 1
7 -> b: 1
7 -> c: 1
7 -> d: -1

8 -> a: -1
8 -> b: -1
8 -> c: -1
8 -> d: 1

9 -> a: 1
9 -> b: -1
9 -> c: -1
9 -> d: 1

Biases:

a: 0
b: 0
c: 0
d: 0

This is to say, if we set the biases all to $0$ and the weights to $-1$ if the bitwise representation of the number uses that digit as $0$ and $1$ if the representation of the number in that digit is $1$, we will get the expected outputs from our network.

### Proving the small step size that maximizes the decrease of C

*Prove the assertion of the last paragraph. Hint: If you're not already familiar with the Cauchy-Schwarz inequality, you may find it helpful to familiarize yourself with it.*

By Cauchy-Schwarz:

$$\langle \Delta v, \grad C \rangle \leq \|\Delta v\| \cdot \|\grad C\|$$

If our objective is to maximize the decrease of $C$, that is, minimize $\Delta C$, then we can equivalently maximize the inner product of its composing vectors and multiply this result by $-1$. By Cauchy-Schwarz, we know this maximum value to be precisely when $\Delta v$ and $\grad C$ are colinear, and since the norm of $\Delta v$ must be $\epsilon$, thus $-\eta$ must be equivalent to $\frac{\epsilon}{\|\grad C\|}$.

### Gradient descent with one variable

*I explained gradient descent when C is a function of two variables, and when it's a function of more than two variables. What happens when C is a function of just one variable? Can you provide a geometric interpretation of what gradient descent is doing in the one-dimensional case?*

In the one-dimensional case, we have exact equivalencies with the two-dimensional case. Geometrically, we are following the derivative's direction until we find the minimum value.

### Online learning

*An extreme version of gradient descent is to use a mini-batch size of just 1. That is, given a training input, $x$, we update our weights and biases according to the rules $w_k \rightarrow w'_k = w_k − \eta \frac{\delta C_x}{\delta w_k}$ and $b_l \rightarrow b'_l = b_l − \eta \frac{\delta C_x}{\delta b_l}$. Then we choose another training input, and update the weights and biases again. And so on, repeatedly. This procedure is known as online, on-line, or incremental learning. In online learning, a neural network learns from just one training input at a time (just as human beings do). Name one advantage and one disadvantage of online learning, compared to stochastic gradient descent with a mini-batch size of, say, $20$.*

**Advantage**: Very easy to compute gradient (20x speedup)
**Disadvantage**: It's possible that we move in the direction opposite the overall gradient (since we would only be using one training input), which would slow down the overall learning.

### Activation layer proof

*Write out Equation (22) in component form, and verify that it gives the same result as the rule (4) for computing the output of a sigmoid neuron.*

The $j$'th entry of $a'$ has the following activation:

$$a' = \sigma (w_j \cdot a_j + b_j)$$

with $w_j \cdot a_j$ being the dot product between the $j$'th row of the weights with the $j$'th layer of outputs. This is exactly equivalent to the output of a sigmoid neuron.

### Two-layer neural network

*Try creating a network with just two layers - an input and an output layer, no hidden layer - with 784 and 10 neurons, respectively. Train the network using stochastic gradient descent. What classification accuracy can you achieve?*

Running the network with layers [784, 10] yielded a max accuracy of $91.68\%% on epoch 28.
