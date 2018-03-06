## Course 2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
### 1주차 (Week 1)
#### Practical aspects of Deep Learning
- Video: Train / Dev / Test sets
	- Applied ML is a highly iterative process
		- \# Layers
		- \# hidden units
		- learning rates
		- activation functions
		- Etc.
		- Iteration of 'Idea, Code, and Experiment'
		- Deep learning has found success in ...
			- NLP
			- Vision
			- Speech
			- Structured Data
				- Ads
				- Search Engine
				- Security
				- Logistics
					- Ex) Figure out where to drop the passenger
		- Someone from NLP might enter Vision field, and someone from Speech field might enter Ads, etc.
		- Intuitions from one domain applications do not usually transfer to other application areas
		- These things depends on amount of data, computer configurations, GPU/CPU, etc.
		- Even the most experienced deep learning people find it almost impossible to correctly guess the best choice of the hyperparameters at the very first time.
		- So, you just need to do iterative process to figure out these things. (hyperparameters)
			- Therefore, how quickly you can go through means how efficiently you can go around the cycle.
	- Train/dev/test sets
		- Data
			- Training set
			- Hold-out cross validation
			<br>(Development Set, a.k.a. "dev" set)
			- Test set
		- Previous Era:
			- 70/30% (Training/Test)
			- 60/20/20% (Training/Dev/Test)
			- ~ 10,000 examples in total
		- Big Data:
			- Usually have around 1,000,000 examples
			- 10,000 is more than enough to test out things for both dev/test sets
			- 98/1/1% or 99.5/.25/.25% works in these days
			<br>(or 99.5/.4/.1%, etc.)
	- Mismatched train/test distribution
		- Example
			- Training: Cat picture from webpages
			- Dev/Test sets: Cat pictures from users using your app
			- Doesn't come from same distribution!
			- Make sure dev & test sets come from the same distribution
				- This way, the progress on your machine learning algorithm will be faster.
		- Not having a test set might be Okay! (Only dev set.)
			- Sometimes, it's training/"test"
			<br>(train/test → train/dev)
- Video: Bias / Variance
	- Bias and Variance
		- high bias: underfitting
		- high varianvce: overfitting
		- Example
			- Considering human error is almost 0%,
			- Train set error 1%, Dev set error 11%
				- It does well on train set, so it's high variance.
			- Train set error 15%, Dev set error 16%
				- It doesn't do well on both sides, so it's underfitting. (high bias)
			- Train set error 15%, Dev set error 30%
				- It has high bias AND high variance. (BOTH)
			- Train set error 0.5%, Dev set error 1%
				- Low bias, low variance
			- optimal (bayes) error is ≒ 0%
				- If optimal error is 15%, then the result with 15% on train/dev set error would be low variance.
	- High bias and high variance
		- High bias in some region and high variance in some region.
- Video: Basic Recipe for Machine Learning
- Video: Regularization
- Video: Why regularization reduces overfitting?
- Video: Dropout Regularization
- Video: Understanding Dropout
- Video: Other regularization methods
- Video: Normalizing inputs
- Video: Vanishing / Exploding Gradients
- Video: Weight Initialization for Deep Networks
- Video: Numerical approximation of gradients
- Video: Gradient checking
- Video: Gradient Checking Implementation Notes
- Quiz: Practical aspects of deep learning
- Programming Assignment: Initialization
- Programming Assignment: Regularization
- Programming Assignment: Gradient Checking
- Video: Yoshua Bengio Interview

### 2주차 (Week 2)
#### Optimization algorithms
- Video: Mini-batch gradient descent
- Video: Understanding mini-batch gradient descent
- Video: Exponentially weighted averages
- Video: Understanding exponentially weighted averages
- Video: Bias correction in exponentially weighted averages
- Video: Gradient descent with momentum
- Video: RMSprop
- Video: Adam optimization algorithm
- Video: Learning rate decay
- Video: The problem of local optima
- Quiz: Optimization algorithms
- Programming Assignment: Optimization
- Video: Yuanqing Lin Interview

### 3주차 (Week 3)
#### Hyperparameter tuning, Batch Normalization and Programming Frameworks
- Video: Tuning process
- Video: Using an appropriate scale to pick hyperparameters
- Video: Hyperparameter tuning in practice: Pandas vs. Caviar
- Video: Normalizing activations in a network
- Video: Fitting Batch Norm into a Neural Network
- Video: Why does Batch Norm work?
- Video: Batch Norm at test time
- Video: Softmax Regression
- Video: Training a softmax classifier
- Video: Deep learning frameworks
- Video: Tensorflow
- Quiz: Hyperparameter tuning, Batch Normalization, Programming Frameworks
- Programming Assignment: Tensorflow