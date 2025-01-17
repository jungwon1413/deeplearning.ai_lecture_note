## Course 1: Neural Networks and Deep Learning
### 1주차 (Week 1)
#### Introduction to deep learning
- Video: Welcome
	- AI is the new electricity
	- Electricity had once transformed countless industries: transportation, manufacturing, healthcare, communications, and more
	- AI will now bring about an equally big transformation.
	- What you'll learn
		- Courses in the sequence (Specialization):
			- Neural Networks and Deep Learning
			- Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
			- Structuring your Machine Learning project
				- Train / Dev / Test
				- End-to-End
			- Convolutional Neural Networks
				- CNN
			- Natural Language Processing: Building sequence models
				- RNN
				- LSTM
- Video: What is a neural network?
	- Housing Price Prediction
		- Neuron
		- ReLU (Rectified Linear Unit)
- Video: Supervised Learning with Neural Networks
	- Supervised Learning - Example
		- Standard NN
		- CNN
		- RNN
		- Custom / Hybrid
	- Neural Network Examples
		- Standard NN
		- Convolutional NN
		- Recurrent NN
	- Supervised Learning
		- Structured data
			- Tabled data
		- Unstructured data
			- Audio
			- Image
			- Text
- Video: Why is Deep Learning taking off?
	- Scale drives deep learning progress
	- Small training data: almost no difference
		- Large training data: significant difference in performance
		- 3 Factors
			- Data
			- Computation
			- Algorithms
		- Iteration of Idea-Code-Experiment
		- 10 min
		- 1 day
		- 1 month
- Video: About this Course
	- Courses in this Specialization
		- Neural Networks and Deep Learning (We're at this step)
		- Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
		- Structuring your Machine Learning project
		- Convolutional Neural Networks
		- Natural Language Processing: Building sequence models
	- Outline of this Course
		- Week 1: Introduction
		- Week 2: Basic Neural Network Programming
		- Week 3: One Hidden Layer Neural Networks
		- Week 4: Deep Neural Networks
- Reading: Frequently Asked Questions
- Video: Course Resources
	- Discussion forum
		- Questions, technical discussions, bug reports, etc.
	- Contact us (deeplearning.ai)
	- Companies (deeplearning.ai)
	- Universities (deeplearning.ai)
- Reading: How to use Discussion Forums
- Quiz: Introduction to deep learning
- Video: Geoffrey Hinton Interview

### 2주차 (Week 2)
#### Neural Networks Basics
- Video: Binary Classification
	- Binary Classification
		- 1 (cat) vs 0 (non cat)
		- Red, Green, and Blue
			- 64 x 64 x 3 = 12,288
			- N = N_x = 12,288
			- X → Y
	- Notation
		- M_train
		- M_test
		- X.shape
		- R_nx x m
- Video: Logistic Regression
	- Logistic Regression
		- Given x, want y_hat = P(y=1 | x)
			- X∈R^nx^, 0 ≤ y_hat ≤ 1
			- Parameters: w∈R_nx, b∈R(real number)
			- Output: y_hat = sigmoid(w^T·x + b)
		- Sigmoid(z) = 1 / (1 + e^(-z))
			- If z large, sigmoid(z) = 1 / (1+0) = 1
			- If z large negative number,<br> sigmoid(z) = 1 / (1 + e^(-z)) ≒ 1 / (1+Big_num) ≒ 0
- Video: Logistic Regression Cost Function
	- Logistic Regression cost function
		- y_hat = sigmoid(w^T·xi + b), where sigmoid(z_i) = 1 / (1 + e^-z_i))
		- Given {(x1, y1), …, (x_m, y_m)}, want  y_hat(i) ≒ yi
		- Loss (error) function:
			- L(y_hat, y) = 1/2·(y_hat - y)^2
			- L(y_hat, y) = -(y log(y_hat) + (1-y) log(1 - y_hat))
				- If y=1: L(y_hat, y) = - log(y_hat)
					- Want log(y_hat) large, want y_hat large.
				- If y=0: L(y_hat, y) = -log(1 - y_hat)
					- Want log(1 - y_hat) … want y_hat small.
		- Cost function:
			- J(w, b) = (1/m) ∑[1∽m](L(y_hat(i), yi)
			- = -(1/m) ∑[1∽m]((yi·log(y_hat(i)) + (1-yi)log(1 - y_hat(i))))
- Video: Gradient Descent
	- Gradient Descent
		- Recap: y_hat = sigmoid(w^T·xi + b), sigmoid(z_i) = 1 / (1 + e^-z_i)
		- J(w, b) = (1/m) ∑[1∽m](L(y_hat(i), yi) = -(1/m) ∑[1∽m]((yi·log(y_hat(i)) + (1-yi)log(1 - y_hat(i))))
		- Want to find w, b, that minimize J(w, b)
		- w := w - α·(dJ(w) / dw)
			- Alpha: learning rate
			- dJ(w) / dw: "dw"
			- Same as… w := w - α·dw
		- J(w, b)
			- w := w - α·(dJ(w, b) / dw)
			- b := b - α·(DJ(w, b) / db)
		- ∂: "partial derivative"<br>d: J
- Video: Derivatives
	- Intuition about derivatives
		- Ex) f(a) = 3a
			- If a = 2, f(a) = 6
			- If a = 2.001, f(a) = 6.003
		- Slope (derivative) of f(a) at a = 2 is 3
			- Slope: height/width
				- 0.003 / 0.001
			- If a = 5, f(a) = 15
			- If a = 5.001, f(a) = 15.003
			- Slope at a = 5 is also 3
- Video: More Derivative Examples
	- Ex) f(a) = a^2
		- If a = 2, f(a) = 4
		- If a = 2.001, f(a) ≒ 4.004 (4.004001)
		- Slope (derivative) of f(a) at a = 2 is 4.
		- (d/da)f(a) = 4 when a = 2
		- If a = 5, f(a) = 25
		- If a = 25, f(a) ≒ 25.010
		- (d/da)f(a) = 10 when a = 5
		- (d/da)f(a) = (d/da) a^2 = 2a
		- (d/da) a^2 = 2a
			- (2a) x 0.001
	- If f(a) = a^2, then (d/da) f(a) = 2a
		- If a = 2, (d/da) f(a) = 4
	- If f(a) = a^3, then (d/da) f(a) = 3a^2
		- If a = 2, (d/da) f(a) = 12
	- If f(a) = log_e(a), then (d/da) f(a) = 1/a
		- If a = 2, (d/da) f(a) = 1/2
- Video: Computation Graph
	- Computation Graph
		- Ex) J(a, b, c) = 3(a + bc)
			- u = bc
			- v = a + bc
				- v = a + u
			- J = 3(a + bc)
				- J = 3v
			- 3(a + bc) = 3(5 + 3x2) = 33
- Video: Derivatives with a Computation Graph
	- Computing derivatives
		- (From previous video) J = 3v
			- v = a + u
			- u = bc
			- a = 5, b = 3, c = 2
		- dJ/dv = ? = 3
			- f(a) = 3a
				- df(a)/da = df/da = 3
			- J = 3v
				- dJ/dv = 3
		- dJ/da = 3 = dJ/dv·dv/da
			- dv/da = 1
			- 3 = 3 x 1
		- J → dJ/dv → dJ/da = "da" = 3
		- d(FindOutputVar)/d(Var)
		- u = bc
			- du = 3
			- dJ/du = 3 = dJ/dv·dv/du = 3·1
			- dJ/db = dJ/du·du/db = 2 = 6
				- dJ/du = 3
				- du/db = 2
			- dJ/dc = dJ/du·du/dc = 9
				- dJ/du = 3
				- du/dc = 3
		- From above, "da" = 3, "db" = 6, "dc" = 9
- Video: Logistic Regression Gradient Descent
	- Logistic regression recap
		- Z = wTx + b
		- y_hat = a = sigmoid(z)
		- L(a, y) = -(y log(a) + (1 - y) log(1 - a))
		- For x1, w1, x2, w2, b
			- Z = w1·x1 + w2·x2 + b → y_hat = a = sigmoid(z) → L(a, y)
	- Logistic regression derivatives
		- L(a, y)<br>→ dL(a, y)/da<br>= "da"<br>= -y/a + (1-y)/(1-a)<br>→ "dz"<br>= dL/dz<br>= dL(a, y)/dz<br>= a - y<br>= dL/da·da/dz<br>= dL/da·a(1-a)
		- ∂L/∂w1 = "dw1" = x1·dz
		- dw2 = x2·dz
		- db = dz
		- In result,
			- w1 := w1 - α·dw1
			- w2 := w2 - α·dw2
			- b := b - α·db
- Video: Gradient Descent on m Examples
	- Logistic regression on m examples
		- (Check slide)
- Video: Vectorization
	- What is vectorization?
		- Non-vectorized:
			- z = 0<br> for i in range(n - x):<br> z +=  w[i]·x[i]<br> z += b
		- Vectorized:
			- z = np.dot(w, x) + b
			- np.dot(w, x): w.T·x
	- Examples in practice
		- A = np.array([1, 2, 3, 4])
			- Result: [1, 2, 3, 4]
		- Vectorized version vs. for loop
			- Vectorized version: 1.5027… ms
			- For loop: 474.2951… ms
	- GPU/CPU - SIMD: Single Instruction Multiple Data
- Video: More Vectorization Examples
	- Neural network programming guideline
		- Whenever possible, avoid explicit for-loops.
			- Bad example
				- u = Av<br> ui = ∑(Ai·v)
				- u = np.zeros((n, 1))<br> for i …:<br> for j …: <br> u[i] += A[i][j]·v[j]
			- Good example
				- u = np.dot(A, v)
	- Vectors and matrix valued functions
		- Say you need to apply the exponential operation on every element of a matrix/vector.
			- u = np.zeros((n, 1))<br>for i in range(n):<br> u[i] = math.exp(v[i])
			- Calculation examples
				- import numpy as np
				- u = np.exp(v)
				- np.log(v)
				- np.abs(v)
				- np.maximum(v, 0)
				- v\*\*2
	- Logistic regression derivatives
		- dw = np.zeros((n_x, 1))
		- dw += xi·dz_i
		- dw /= m
- Video: Vectorizing Logistic Regression
	- Vectorizing Logistic Regression
		- First set
			- z(1) = w^T·x(1) + b
			- a(1) = sigmoid(z(1))
		- Second set
			- z(2) = w^T·x(2) + b
			- a(1) = sigmoid(z(2))
		- Third set
			- z(3) = w^T·x(3) + b
			- a(1) = sigmoid(z(3))
		- Vectorized calculation
			- z = np.dot(w.T, x) + b
				- "broadcasting"
				- [z(1), z(2), ..., z(m)] = w^T·x + [b(1), b(2), ..., b(m)]
			- A = [a(1), a(2), ..., a(m)] = sigmoid(z)
- Video: Vectorizing Logistic Regression's Gradient Output(Computation)
	- Vectorizing Logistic Regression
		- (Check slide)
	- Implementing Logistic Regression
		- for iter in range(1000):
			- z = np.dot(w.T, X) + b
			- A = sigmoid(z)
			- dZ = A - Y
			- dw = 1/m·X·dZT
			- db = 1/m·np.sum(dZ)
			- w := w - α·dw
			- b := b - α·db
- Video: Broadcasting in Python
	- Broadcasting example
		- Cal = A.sum(axis=0)
			- [59.   239.    155.4    76.9]
		- 100·A/Cal.reshape(1, 4)
			- [[94.91525424    0.          …    ]<br> [   2.03389831    43.51…       ]<br> [ … ]]
			- A / Cal
				- (3, 4) / (1, 4)
	- General Principle
		- (m, n) & [+, -, \*, /]
			- (1, n) → (m, n)
			- (m, 1) → (m, n) 
		- Matlab / Octave: bsxfun
- Video: A note on python/numpy vectors
	- Python Demo
	- Python / numpy vectors
		- a = np.random.randn(5)
			- a.shape = (5, )
			- "rank 1 array"
			- DON'T USE!
		- a = np.random.randn(5, 1)
			- a.shape = (5, 1)
			- Column vector
		- a = np.random.randn(1, 5)
			- a.shape = (1, 5)
			- Row vector
		- assert(a.shape == (5, 1))
			- a = a.shape((5,1))
- Video: Quick tour of Jupyter/iPython Notebooks
- Video: Explanation of logistic regression cost function (optional)
	- Logistic regression cost function
		- If y = 1: p(y|x) = y_hat
		- If y = 0: p(y|x) = 1 - y_hat
		- To sum up, it's p(y|x)
		- p(y|x) = y_haty (1 - y_hat)(1 - y)
			- If y = 1: p(y|x) = y_hat (1 - y_hat)0
			- If y = 0: p(y|x) = y_hat0 (1 - y_hat)1<br>=  1 x (1 - y_hat) = 1 - y_hat
			- Log p(y|x) = log(y_hat) (1 - y_hat)(1-y)<br>= y log(y_hat) + (1 - y) log(1 - y_hat)<br>= - L(y_hat, y)
	- Cost on m examples
		- log p(labels in training set) = log∏[1∽m]p(yi|xi)
		- log p(…) = ∑[1∽m] log p(yi|xi)<br>(log p(yi|xi) = -L(y_hat(i)|yi) )<br>= -∑[1∽m]L(y_hat(i)|yi)
		- Cost (to minimize)
			- J(w, b) = 1/m ∑[1∽m]L(y_hat(i)|yi)
		- "Maximum likelihood estimation"
- Quiz: Neural Network Basics
- Reading: Deep Learning Honor Code
- Reading: Programming Assignment FAQ
- Practice Programming Assignment: Python Basics with numpy (optional)
- Programming Assignment: Logistic Regression with a Neural Network mindset
- Video: Pieter Abbeel Interview


### 3주차 (Week 3)
#### Shallow Neural Networks
- Video: Neural Networks Overview
	- What is a Neural Network?
		- Single Neuron Computation
			- z = W^T·x + b
				- z becomes dz when backpropagate
			- a = sigmoid(z)
				- a becomes dz when backpropagate
			- L(a,y)
		- Multiple Neuron Computation
			- z[1] = w[1]·x + b[1]
			- a[1] = sigmoid(z[1])
				- Calculate dw[1] & db[1]
			- z[2] = W[2]·a[1] + b[2]
				- = dz[2]
			- a[2] = sigmoid(z[2])
			- L(a[2], y)
- Video: Neural Network Representation
	- Neural Network Representation
		- x1, x2, x3, ...: Input layer
		- Layer with Computational Neurons: Hidden layer
		- Layer with making result: Output layer
		- a[0] = X (Input layer is usually 0th layer)
			- Count from first hidden layer to last hidden layer: number of layers
		- a[1] = [a1[1], a2[1], a3[1], a4[1], ...]
			- Has W[1] & b[1] values to compute
				- If W[1] has (4, 3): 4 neurons at 1st layer & 3 input values
				- If b[1] has (4, 1): 4 neurons at 1st layer
		- a[2]: Sum all up at output layer
			- Has W[2] & b[2] values (assuming it's 2 layer neural network)
				- If W[2] has (1, 4): 1 neuron at 2nd layer & 4 inputs from previous layer (1st layer)
				- If b[2] has (1, 1): 1 neuron at 2nd layer
		- If 2 layer network,
			- y_hat = a[2]
			- "y_hat = a"
- Video: Computing a Neural Network's Output
	- Neural Network Representation
		- Single Neural Network (1 neuron)
			- z = w^T·x + b
			- a = sigmoid(z)
		- Multi Neural Network (many neurons)
			- 1st cell
				- z1[1] = w1[1]^T·x + b1[1]
				- a1[1] = sigmoid(z1[1])
			- 2nd cell
				- z2[1] = w2[1]^T·x + b2[1]
				- a2[1] = sigmoid(z2[1])
			- And so on (for rest of the remaining cells)
			- W[1] = [w1[1], w2[1], w3[1], ...]
			- X = [x1, x2, x3, ...]
			- b[1] = [b1[1], b2[1], b3[1], ...]
			- Then we need to calculate z[1]
				- z[1] = [z1[1], z2[1], z3[1], ...]
			- To sum up
				- z[1] = W[1]^T·x + b[1]
			- a[1] = [a1[1], a2[1], a3[1], ...]
				- = sigmoid(z[1])
	- Neural Network Representation learning
		- Assuming there's single layer NN which has 3 inputs, 4 neurons, and 1 output
			- x = a[0]
			- z[1] = w[1]·x + b[1]
				- (4,1) = (4,3)·(3,1) + (4,1)
			- a[1] = sigmoid(z[1])
				- (4,1) = (4,1)
			- z[2] = W[2]·a[1] + b[2]
				- (1,1) = (1,4)·(4,1) + (1,1)
			- a[2] = sigmoid(z[2])
				- (1,1) = (1,1)
- Video: Vectorizing across multiple examples
	- Vectorizing across multiple examples
		- From previous lecture...
			- z[1] = w[1]·x + b[1]
			- a[1] = sigmoid(z[1])
			- z[2] = W[2]·a[1] + b[2]
			- a[2] = sigmoid(z[2])
		- x → a[2] = y_hat
		- x(1) → a[2]\(1) = y_hat(1)
		- x(2) → a[2]\(2) = y_hat(2)
		- x(3) → a[2]\(3) = y_hat(3)
		- And so on... until,
		- x(m) → a[2]\(m) = y_hat(m)
			- a[2]\(i)
				- [2]: layer 2
				- (i): example i
		- We can compute values by this way:
			- for i = to m:
				- z[1]\(i) = w[1]·x(i) + b[1]
				- a[1]\(i) = sigmoid(z[1]\(i))
				- z[2]\(i) = W[2]·a[1]\(i) + b[2]
				- a[2]\(i) = sigmoid(z[2]\(i))
		- X = [x(1), x(2), x(3), ..., x(m)] \(and it has (nx, m) shape)
		- Then we can use vectorized method
			- z[1] = W[1]·X + b[1]
				- z[1] = [z[1]\(1), z[1]\(2), z[1]\(3), ..., z[1]\(m)]
				- Shape: (training examples, hidden units)
			- A[1] = sigmoid(z[1])
				- A[1] = [a[1]\(1), a[1]\(2), a[1]\(3), ..., a[1]\(m)]
				- Shape: (training examples, hidden units)
			- z[2] = W[2]·A[1] + b[2]
			- A[2] = sigmoid(z[2])
- Video: Explanation for Vectorized Implementation
	- Justification for vectorized implementation
		- Example
			- z[1]\(1) = w[1]·x(1) + ~b[1]~ 0 (Just to simplify the calculation)
			- z[1]\(2) = w[1]·x(2) + ~b[1]~ 0 (Same as above)
			- z[1]\(3) = w[1]·x(3) + ~b[1]~ 0 (Same as above)
			- w[1]·x(1): Will have (training examples, 1) shape (1st column or feature)
			- w[1]·x(2): Will have (training examples, 1) shape (2nd column or feature)
			- w[1]·x(3): Will have (training examples, 1) shape (3rd column or feature)
			- Then, it can be expressed as following:
				- W[1]·X = W[1]·[x(1), x(2), x(3), ...] \(training examples)
				<br>= [z[1]\(1), z[1]\(2), z[1]\(3), ..., z[1]\(m)]
				<br>= z[1]
				- So, z[1] = W[1]·X + b[1]
	- Recap of vectorizing across multiple examples
		- 1st method (non-vectorized)
			- for i=1 to m:
				- z[1]\(i) = w[1]·x(i) + b[1]
				- a[1]\(i) = sigmoid(z[1]\(i))
				- z[2]\(i) = W[2]·a[1]\(i) + b[2]
				- a[2]\(i) = sigmoid(z[2]\(i))
		- 2nd method (vectorized)
			- z[1] = W[1]·X + b[1]
				- X: A[0] \(since x = a[0])
				- same as W[1]·A[0] + b[1]
			- a[1] = sigmoid(z[1])
			- z[2] = W[2]·A[1]) + b[2]
			- a[2] = sigmoid(z[2])
- Video: Activation Functions
	- Activation functions
		- From previous lecture ... (with some fixes)
			- z[1] = W[1]·X + b[1]
			- a[1] = ~sigmoid(z[1])~ g(z[1])
				- where 'g' can be non-linear function
			- z[2] = W[2]·A[1]) + b[2]
			- a[2] = ~sigmoid(z[2])~ g(z[2])
		- Different kinds of non-linear function
			- sigmoid function:
				- a = 1 / (a + e^-2)
				- This goes between 0 and 1
				- Centered at 0.5
			- tanh function:
				- a = tanh(z)
				<br> (e^z - e^-z) / (e^z + e^-z)
				- This goes between -1 and 1
				- Centered at 0
				- Almost always works better than sigmoid function
			- ReLU function:
				- a = max(0, z)
				- ReLU: Rectified Linear Unit
			- Leaky ReLU function:
				- -0.x if less than 0, z if equal or greater than 0
				- Common multiplier for negatives are 0.01
		- There's various ways to use activation functions
			- Examples
			    - Use tanh for hidden Layers
			    - Use sigmoid for output layer
			    	- You want results to be within y∈{0, 1} range
	- Pros and cons of activation functions
- Video: Why do you need non-linear activation functions?
	- Activation function
		- Recall from last lecture,
			- z[1] = W[1]·X + b[1]
			- a[1] = ~g[1]\(z[1])~ z[1]
				- g(z) = z
				- It's "linear activation function"
			- z[2] = W[2]·A[1]) + b[2]
			- a[2] = ~g[2]\(z[2])~ z[2]
		- Linear function calculation
			- a[1] = z[1] = W[1]·X + b[1]
			- a[2] = z[2] = W[2]·a[1] + b[2]
			<br>= W[2]·(W[1]·X + b[1]) + b[2] where (W[1]·X + b) = a[1]
			- (W[2]·W[1])·X + (W[2]·b[1] + b[2])
				- W[2]·W[1] = W'
				- W[2]·b[1] + b[2] = b'
			- = W'·X + b
			<br> where g(z) = z
			- It's just a linear calculation, without any significant purpose.
- Video: Derivatives of activation functions
	- Sigmoid activation function
		- d/dz·g(z): slope of ~g(x)~ g(z) at z
		<br>= 1/(1 + e^-z) \* (1 - 1/(1 + e^-z))
		<br>= g(z)(1 - g(z))
		<br>= g'(z)
		- Therefore, g'(z) = a·(1 - a)
		- Some examples
			- z=10, g(z)≒1
				- d/dz·g(z) ≒ 1(1 - 1) ≒ 0
			- z=-10, g(z)≒0
				- d/dz·g(z) ≒ 0(1 - 0) ≒ 0
			- z=0, g(z)≒1/2
				- d/dz·g(z) ≒ 1/2(1 - 1/2) = 1/4
	- Tanh activation function
		- g(z) = tanh(z)
		<br>= (e^z - e^-z) / (e^z + e^-z)
		- d/dz·g(z): slope of g(z) at z
		<br>= 1 - (tanh(z))^2
		- a = g(z)
		<br>g'(z) = 1 - a^2
		- z=10, tanh(z)≒1
			- g'(z) ≒ 0
		- z=-10, tanh(z)≒-1
			- g'(z) ≒ 0
		- z=0, tanh(z)≒0
			- g'(z) ≒ 1
	- ReLU and Leaky ReLU
		- ReLU
			- g(z) = max(0, z)
			- g'(z) =
			<br>0 if z < 0
			<br>1 if z ≥ 0
			<br>~undefined if z=0~ (ignore this case)
		- Leaky ReLU
			- g(z) = max(0.01z, z)
			- g'(z) =
			<br>0.01 if z < 0
			<br>1 if z ≥ 0
- Video: Gradient Descent of Neural Networks
	- Gradient descent for neural networks
		- Parameters: W[1], b[1], W[2], b[2]
		- Shapes
			- W[1]: (n[1], n[0])
			- b[1]: (n[1], 1)
			- W[2]: (n[2], n[1])
			- b[2]: (n[1], 1)
		- Cost function: J(W[1], b[1], W[2], b[2])
		<br>1/m·∑[1∽m]L(y_hat, y) (where y_hat = a[2])
		- Gradient descent:
			- Repeat following,
				- Compute predicts (y_hat(i), i = 1, ..., m)
				- dw[1] = dJ/dW[1]
				- db[1] = ∂J/∂b[1]
				- ...
				- W[1] := W[1] - α·dW[1]
				- b[1] := b[1] - α·db[1]
				- ...
	- Formulas for computing derivatives
		- Forward propagation:
			- z[1] = W[1]·X + b[1]
			- A[1] = g[1]\(z[1])
			- z[2] = W[2]·A[1] + b[2]
			- A[2] = g[2]\(z[2]) = sigmoid(z[2])
		- Backpropagation:
			- dz[2] = A[2] - Y (from loss function)
			- dW[2] = 1/m · dz[2] · A[1]^T
			- db[2] = 1/m · np.sum(dz[2], axis=1, keepdims=True)
				- axis=1: summing horizontally
				- keepdims=True: prevents from 'rank 1 arrays'
					- rank 1 array: (n[2], )
					- This becomes (n[2], 1)
			- dz[1] = W[2]^T · dz[2] \* g'[1]\(z[1])
				- \*: element-wise product
				- W[2]^T · dz[2] has (n[1], m) shape
				- g'[1]\(z[1]) has (n[1], m) shape
			- dW[1] = 1/m · dz[1] · X^T
			- db[1] = 1/m · np.sum(dz[1], axis=1, keepdims=True)
- Video: Backpropagation Intuition (optional)
	- Computing gradients
		- Logistic regression
			- d/da · L(a, y)
			<br>= -y·log(a) - (1-y)·log(1-a)
			<br>= -y/a + (1-y)/(1-a)
			<br>= da
			- dz
			<br>= a - y
			<br>= da · g'(z) (where g(z) = sigmoid(z))
			<br> ∂L/∂z = ∂L/∂a · da/dz
				- da/dz
				<br>= d/dz · g(z)
				<br>= g'(z)
				- "dz" = da
			- dw = dz·x
			- db = dz
	- Neural network gradients
		- Example
			- Layer 2 has W[2], b[2] when forward propagate
			- Layer 2 has dW[2], db[2] when backpropagate
		- dz[2] = a[2] - y
		- dW[2] = dz[2]·a[1]^T → "dw = dz·X"
		- db[2] = dz[2]
		- dz[1] = W[2]^T · dz[2]
		<br>\* g'[1]\(z[1])
	- Summary of gradient descent
		- dz[2] = a[2] - y
		- dW[2] = dz[2]·a[1]^T
		- db[2] = dz[2]
		- dz[1] = W[2]^T·dz[2] \* g'[1]\(z[1])
		- dW[1] = dz[1]·X^T
		- db[1] = dz[1]
		- Vectorized Implementation (forward propagation)
			- (by element) z[1] = w[1]·x + b[1]
			- (by element) a[1] = g[1]\(z[1])
			- z[1] = [z[1]\(1), z[1]\(2), ..., z[1]\(m)]
			- z[1] = W[1]·X + b[1]
			- A[1] = g[1]\(z[1])
		- Vectorized Implentation (backpropagation, pseudo code)
			- dZ[2] = A[2] - Y
			- dW[2] = 1/m·dZ[2]·A[1]^T
			- db[2] = 1/m·np.sum(dZ[2], axis=1, keepdims=True)
			- dZ[1] = W[2]^T·dZ[2] \* g'[1]\(Z[1])
				- (shape) dZ[1]: (n[1], m)
				- (shape) W[2]^T·dZ[2]: (n[1], m)
				- (shape) g'[1]\(Z[1]): (n[1], m)
			- dW[1] = 1/m·dZ[1]·X^T
			- db[1] = 1/m·np.sum(dZ[1], axis=1, keepdims=True)
- Video: Random Initialization
	- What happens if you initialize weights to zero?
		- W[1] =
		<br>[0 0
		<br>0 0]
		- b[1] =
		<br>[0
		<br>0]
		- a1[1] = a2[1]
		- dz1[1] = dz2[1]
		- W[2] = [0 0]
		- "Symmetric"
		- W[1] := W[1] - α·dW
			- Resulting that the 1st row will be equal to the 2nd row
			- Thus, it's same as having 1 neuron
	- Random initialization
		- W[1] = np.random.randn((2,2)) \* 0.01
			- Generates 'gaussian random variable'
			- 0.01 is multiplied to produce very small number to initiate
				- Why not 100?
					- This will result sigmoid/tanh function to be extremely large value
					- Having large value weight will result slow learning rate
				- Sometimes, there can be better value than 0.01
					- However, if you train very DEEP neural network, you might want to change this value
				- However, 0.01 will probably work OK
		- b[1] = np.zeros((2,1))
			- b does not have the symmetry problem
		- W[2] = np.random.randn((1,2)) \* 0.01
		- b[2] = 0 (zero initialization)
- Quiz: Shallow Neural Networks
- Programming Assignment: Planar data classification with a hidden layer
- Video: Ian Goodfellow Interview

### 4주차 (Week 4)
#### Deep Neural Networks
- Video: Deep L-layer neural network
	- What is a deep neural network?
		- logistic regression (1 layer NN)
			- Input, single neuron, and output
			- This is an example of "shallow" network
		- 1 hidden layer
			- Input, 1 layer with multiple neurons, and output
		- 2 hidden layer (2 layer NN)
			- Input, 2 layers with multiple neurons (both), and output
		- 5 hidden layer
			- Input, 5 layers with multiple neurons, and output
			- People usually call "deep" neural network when a network has more than 5 layers
	- Deep neural network notation
		- *l* = 4 (number of layers)
			- 'This NN has 4 layers'
		- n[*l*] = number of units in layer *l*
		- Input layers are 'layer 0'
			- x = a[0]
		- layer count includes hidden & output layers
		- a[*l*] = activations in layer *l*
			- a[*l*] = g[*l*]\(z[*l*])
		- W[*l*] = weights for z[*l*]
- Video: Forward Propagation in a Deep Network
	- x:
		- z[1] = w[1]·x + b[1]
			- = w[1]·a[0] + b[1]
		- a[1] = g[1]\(z[1])
		- z[2] = w[2]·a[1] + b[2]
		- a[2] = g[2]\(z[2])
		- ...
	- z[*l*] = W[*l*]·a[*l* - 1] + b[*l*]
		- Vectorized: Z[*l*] = W[*l*]·A[*l* - 1] + b[*l*]
			- A[0] = X
	- a[*l*] = g[*l*]\(z[*l*])
		- Vectorized: A[*l*] = g[*l*]\(Z[*l*])
	- Vectorized method
		- Z[1] = W[1]·X + b[1]
			- = W[1]·A[0] + b[1]
		- A[1] = g[1]\(z[1])
		- Z[2] = W[2]·A[1] + b[1]
		- A[2] = g[2]\(Z[2])
		- ...
		- Y_hat = g(Z[4]) = A[4]
- Video: Getting your matrix dimensions right
	- Parameters W[*l*] and b[*l*]
		- z[1] = w[1]·x + b[1]
			- (shape) z[1]: (n[1], 1)
			- (shape) w[1]: (n[1], n[0])
			- (shape) x: (n[0], 1)
			- (shape) b[1]: (n[1], 1)
		- W[*l*]: (n[*l*], n[*l* - 1])
		- b[*l*]: (n[*l*], 1)
		- dW[*l*]: (n[*l*], n[*l* - 1])
		- db[*l*]: (n[*l*], 1)
	- Vectorized implementation
		- Z[1] = W[1]·X + b[1]
			- (shape) Z[1]: (n[1], m)
			- (shape) W[1]: (n[1], n[0])
			- (shape) X: (n[0], m)
			- (shape) b[1]: (n[1], 1) → (n[1], m)
		- z[*l*], a[*l*]: (n[*l*], 1)
		- Z[*l*], A[*l*]: (n[*l*], m)
			- l=0, A[0] = X = (n[0], m)
		- dZ[*l*], dA[*l*]: (n[*l*], m)
- Video: Why deep representations?
	- Intuition about deep representation
		- Earlier layer: represent "simple" features (ex: edge)
		- Later layer: represent "complex" features (ex: human face, fur details, etc.)
		- Audio:
		<br>low level audio waveform features (maybe)
		<br>→ Phonemes (C A T) → Words → Sentence/Phrases
	- Circuit theory and deep learning
		- Informally: There are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute.
		- y = x1 (XOR) x2 (XOR) x3 (XOR) ... (XOR) xn
			- O(log n)
		- You can compute anything with a "single" layer,
			- But you will have exponentially large number of neurons, compared to multi-layer neural network
			- 2^(n-1) neurons will be required
			- O(2^n)
- Video: Building blocks of deep neural networks
	- Forward and backward functions
		- Layer *l*: W[*l*], b[*l*]
		- Forward: Input a[*l* - 1], Output a[*l*]
			- z[*l*] = W[*l*]·a[*l* - 1] + b[*l*]
				- Output a[*l*]
				- Cache z[*l*]
			- a[*l*] = g[*l*]\(z[*l*])
		- Backward: Input da[*l*], Output da[*l* - 1]
			- da[*l*]: Cache z[*l*]
			- da[*l* - 1]: dW[*l*], db[*l*]
		- W[*l*] := W[*l*] - α·dW[*l*]
		- b[*l*] := b[*l*] - α·db[*l*]
	- (Check slide for more details)
- Video: Forward and Backward Propagation
	- Forward propagation for layer *l*
		- Input a[*l* - 1]
		- Output a[*l*], cache (z[*l*])
			- z[*l*]: W[*l*], b[*l*]
		- z[*l*] = W[*l*]·a[*l* - 1] + b[*l*]
		- a[*l*] = g[*l*]\(z[*l*])
		- Vectorized:
			- Z[*l*] = W[*l*]·A[*l* - 1] + b[*l*]
			- A[*l*] = g[*l*]\(Z[*l*])
	- Backward propagation for layer *l*
		- Input da[*l*]
		- Output da[*l* - 1], dW[*l*], db[*l*]
		- dz[*l*] = da[*l*] \* g'[*l*]\(z[*l*])
		- dW[*l*] = dz[*l*]·a[*l* - 1]
		- db[*l*] = dz[*l*]
		- da[*l* - 1] = W[*l*]^T·dz[*l*]
			- dz[*l*] = W[*l* + 1]^T·dz[*l* + 1] \* g'[*l*]\(z[*l*])
		- Vectorized:
			- dZ[*l*] = dA[*l*] \* g'[*l*]\(Z[*l*])
			- dW[*l*] = 1/m·dZ[*l*]·A[*l* - 1]^T
			- db[*l*] = 1/m·(dZ[*l*], axis=1, keepdims=True)
			- dA[*l* - 1] = W[*l*]^T·dZ[*l*]
	- Summary
- Video: Parameters vs Hyperparameters
	- What are hyperparameters?
		- Parameters: W[1], b[1], W[2], b[2], W[3], b[3], ...
		- Hyperparameters:
			- learning rate (α)
			- Number of iterations
			- Number of hidden layers (L)
			- Number of hidden units (n[1], n[2], ...)
			- Choice of activation function
			- More ... (discussed at the later lectures)
				- Momentum
				- Minibatch size
				- Regularization Parameters
	- Applied deep learning is a very empirical process
		- Iteration of 'Idea, Code, and Experiment'
		- Change learning rate, etc.
		- Applied deep learning to ...
			- Vision
			- Speech/NLP
			- Advertising
			- Web Search
			- Recommendation
- Video: What does this have to with the brain?
	- Forward and backward propagation
		- "It's like the brain."
			- It's probably over-simplified.
		- Structure of the linear regression is inspired by brain neuron
			- Axon
- Quiz: Key concepts on Deep Neural Networks
- Programming Assignment: Building your Deep Neural Network: Step by Step
- Programming Assignment: Deep Neural Network Application