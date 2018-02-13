# deeplearning.ai_lecture_note
Summary of deeplearning.ai (Coursera's deep learning specialization course)<br>
<br>
## Course 1: Neural Networks and Deep Learning<br>
### 1주차<br>
Introduction to deep learning<br>
	- Video: Welcome<br>
			§ AI is the new electricity<br>
			§ Electricity had once transformed countless industries: transportation, manufacturing, healthcare, communications, and more<br>
			§ AI will now bring about an equally big transformation.<br>
		○ What you'll learn<br>
			§ Courses in the sequence (Specialization):<br>
				□ Neural Networks and Deep Learning<br>
				□ Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization<br>
				□ Structuring your Machine Learning project<br>
					® Train / Dev / Test<br>
					® End-to-End<br>
				□ Convolutional Neural Networks<br>
					® CNN<br>
				□ Natural Language Processing: Building sequence models<br>
					® RNN<br>
					® LSTM<br>
	- Video: What is a neural network?<br>
		○ Housing Price Prediction<br>
			§ Neuron<br>
			§ ReLU (Rectified Linear Unit)<br>
	- Video: Supervised Learning with Neural Networks<br>
		○ Supervised Learning - Example<br>
			§ Standard NN<br>
			§ CNN<br>
			§ RNN<br>
			§ Custom / Hybrid<br>
		○ Neural Network Examples<br>
			§ Standard NN<br>
			§ Convolutional NN<br>
			§ Recurrent NN<br>
		○ Supervised Learning<br>
			§ Structured data<br>
				□ Tabled data<br>
			§ Unstructured data<br>
				□ Audio<br>
				□ Image<br>
				□ Text<br>
	- Video: Why is Deep Learning taking off?<br>
		○ Scale drives deep learning progress<br>
			§ Small training data: almost no difference<br>
			§ Large training data: significant difference in performance<br>
			§ 3 Factors<br>
				□ Data<br>
				□ Computation<br>
				□ Algorithms<br>
			§ Iteration of Idea-Code-Experiment<br>
				□ 10 min<br>
				□ 1 day<br>
				□ 1 month<br>
	- Video: About this Course<br>
		○ Courses in this Specialization<br>
			§ Neural Networks and Deep Learning (We're at this step)<br>
			§ Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization<br>
			§ Structuring your Machine Learning project<br>
			§ Convolutional Neural Networks<br>
			§ Natural Language Processing: Building sequence models<br>
		○ Outline of this Course<br>
			§ Week 1: Introduction<br>
			§ Week 2: Basic Neural Network Programming<br>
			§ Week 3: One Hidden Layer Neural Networks<br>
			§ Week 4: Deep Neural Networks<br>
	- Reading: Frequently Asked Questions<br>
	- Video: Course Resources<br>
		○ Discussion forum<br>
			§ Questions, technical discussions, bug reports, etc.<br>
		○ Contact us (deeplearning.ai)<br>
		○ Companies (deeplearning.ai)<br>
		○ Universities (deeplearning.ai)<br>
	- Reading: How to use Discussion Forums<br>
	- Quiz: Introduction to deep learning<br>
	- Video: Geoffrey Hinton Interview<br>
<br>
### 2주차<br>
Neural Networks Basics<br>
	- Video: Binary Classification<br>
		○ Binary Classification<br>
			§ 1 (cat) vs 0 (non cat)<br>
			§ Red, Green, and Blue<br>
				□ 64 x 64 x 3 = 12,288<br>
				□ N = Nx = 12,288<br>
				□ X → Y<br>
		○ Notation<br>
			§ M_train<br>
			§ M_test<br>
			§ X.shape<br>
			§ Rnx x m<br>
	- Video: Logistic Regression<br>
		○ Logistic Regression<br>
			§ Given x, want y_hat = P(y=1 | x)<br>
				□ X∈Rnx, 0≤y_hat≤1<br>
				□ Parameters: w∈Rnx, b∈R(real number)<br>
				□ Output: y_hat=sigmoid(wT * x + b)<br>
			§ Sigmoid(z) = 1 / (1 + e-z)<br>
				□ If z large, sigmoid(z) = 1 / (1+0) = 1<br>
				□ If z large negative number,sigmoid(z) = 1 / (1 + e-z) ≒ 1 / (1+Big_num) ≒ 0<br>
	- Video: Logistic Regression Cost Function<br>
		○ Logistic Regression cost function<br>
			§ y_hat = sigmoid(wT * x(i) + b), where sigmoid(z(i)) = 1 / (1 + ez(i))<br>
			§ Given {(x(1), y(1)), …, (x(m), y(m))}, want  y_hat(i) ≒ y(i)<br>
			§ Loss (error) function:<br>
				□ L(y_hat, y) = 1/2 * (y_hat - y)2<br>
				□ L(y_hat, y) = -(y*log(y_hat) + (1-y)log(1 - y_hat))<br>
					® If y=1: L(y_hat, y) = -log(y_hat)<br>
						◊ Want log(y_hat) large, want y_hat large.<br>
					® If y=0: L(y_hat, y) = -log(1 - y_hat)<br>
						◊ Want log(1 - y_hat) … want y_hat small.<br>
			§ Cost function:<br>
				□ J(w, b) = (1/m) ∑1~m(L(y_hat(i), y(i))<br>
				□ = -(1/m) ∑1~m((y(i)*log(y_hat(i)) + (1-y(i))log(1 - y_hat(i))))<br>
	- Video: Gradient Descent<br>
		○ Gradient Descent<br>
			§ Recap: y_hat = sigmoid(wT * x(i) + b), sigmoid(z(i)) = 1 / (1 + e-z(i))<br>
			§ J(w, b) = (1/m) ∑1~m(L(y_hat(i), y(i))= -(1/m) ∑1~m((y(i)*log(y_hat(i)) + (1-y(i))log(1 - y_hat(i))))<br>
			§ Want to find w, b, that minimize J(w, b)<br>
			§ w := w - alpha * (dJ(w) / dw)<br>
				□ Alpha: learning rate<br>
				□ dJ(w) / dw: "dw"<br>
				□ Same as… w := w - alpha * dw<br>
			§ J(w, b)<br>
				□ w := w - alpha * (dJ(w, b) / dw)<br>
				□ b := b - alpha * (DJ(w, b) / db)<br>
			§ ∂: "partial derivative"d: J<br>
	- Video: Derivatives<br>
		○ Intuition about derivatives<br>
			§ Ex) f(a) = 3a<br>
				□ If a = 2, f(a) = 6<br>
				□ If a = 2.001, f(a) = 6.003<br>
			§ Slope (derivative) of f(a) at a = 2 is 3<br>
				□ Slope: height/width<br>
					® 0.003 / 0.001<br>
				□ If a = 5, f(a) = 15<br>
				□ If a = 5.001, f(a) = 15.003<br>
				□ Slope at a = 5 is also 3<br>
	- Video: More Derivative Examples<br>
		○ Ex) f(a) = a2<br>
			§ If a = 2, f(a) = 4<br>
			§ If a = 2.001, f(a) ≒ 4.004 (4.004001)<br>
			§ Slope (derivative) of f(a) at a = 2 is 4.<br>
			§ (d/da)f(a) = 4 when a = 2<br>
			§ If a = 5, f(a) = 25<br>
			§ If a = 25, f(a) ≒ 25.010<br>
			§ (d/da)f(a) = 10 when a = 5<br>
			§ (d/da)f(a) = (d/da) a2 = 2a<br>
			§ (d/da) a2 = 2a<br>
				□ (2a) x 0.001<br>
		○ If f(a) = a2, then (d/da)f(a) = 2a<br>
			§ If a = 2, (d/da)f(a) = 4<br>
		○ If f(a) = a3, then (d/da)f(a) = 3a2<br>
			§ If a = 2, (d/da)f(a) = 12<br>
		○ If f(a) = loge(a), then (d/da)f(a) = 1/a<br>
			§ If a = 2, (d/da)f(a) = 1/2<br>
	- Video: Computation Graph<br>
		○ Computation Graph<br>
			§ Ex) J(a, b, c) = 3(a + bc)<br>
				□ u = bc<br>
				□ v = a + bc<br>
					® v = a + u<br>
				□ J = 3(a + bc)<br>
					® J = 3v<br>
				□ 3(a + bc) = 3(5 + 3x2) = 33<br>
	- Video: Derivatives with a Computation Graph<br>
		○ Computing derivatives<br>
			§ (From previous video) J = 3v<br>
				□ v = a + u<br>
				□ u = bc<br>
				□ a = 5, b = 3, c = 2<br>
			§ dJ/dv = ? = 3<br>
				□ f(a) = 3a<br>
					® df(a)/da = df/da = 3<br>
				□ J = 3v<br>
					® dJ/dv = 3<br>
			§ dJ/da = 3 = dJ/dv * dv/da<br>
				□ dv/da = 1<br>
				□ 3 = 3 x 1<br>
			§ J → dJ/dv → dJ/da = "da" = 3<br>
			§ d(FindOutputVar)/d(Var)<br>
			§ u = bc<br>
				□ du = 3<br>
				□ dJ/du = 3 = dJ/dv * dv/du = 3 * 1<br>
				□ dJ/db = dJ/du * du/db = 2 = 6<br>
					® dJ/du = 3<br>
					® du/db = 2<br>
				□ dJ/dc = dJ/du * du/dc = 9<br>
					® dJ/du = 3<br>
					® du/dc = 3<br>
			§ From above, "da" = 3, "db" = 6, "dc" = 9<br>
	- Video: Logistic Regression Gradient Descent<br>
		○ Logistic regression recap<br>
			§ Z = wTx + b<br>
			§ y_hat = a = sigmoid(z)<br>
			§ L(a, y) = -(ylog(a) + (1 - y)log(1 - a))<br>
			§ For x1, w1, x2, w2, b<br>
				□ Z = w1x1 + w2x2 + b → y_hat = a = sigmoid(z) → L(a, y)<br>
		○ Logistic regression derivatives<br>
			§ L(a, y)→ dL(a, y)/da    = "da"    = -y/a + (1-y)/(1-a)→ "dz"    = dL/dz    = dL(a, y)/dz    = a - y    = dL/da * da/dz    = dL/da * a(1-a)<br>
			§ ∂L/∂w1 = "dw1" = x1 * dz<br>
			§ dw2 = x2 * dz<br>
			§ db = dz<br>
			§ In result,<br>
				□ w1 := w1 - alpha * dw1<br>
				□ w2 := w2 - alpha * dw2<br>
				□ b := b - alpha * db<br>
	- Video: Gradient Descent on m Examples<br>
		○ Logistic regression on m examples<br>
			§ (Check slide)<br>
	- Video: Vectorization<br>
		○ What is vectorization?<br>
			§ Non-vectorized:<br>
				□ z = 0for i in range(n - x):    z +=  w[i] * x[i]z += b<br>
			§ Vectorized:<br>
				□ z = np.dot(w, x) + b<br>
				□ np.dot(w, x): wTx<br>
		○ Examples in practice<br>
			§ A = np.array([1, 2, 3, 4])<br>
				□ Result: [1, 2, 3, 4]<br>
			§ Vectorized version vs. for loop<br>
				□ Vectorized version: 1.5027… ms<br>
				□ For loop: 474.2951… ms<br>
		○ GPU/CPU - SIMD: Single Instruction Multiple Data<br>
	- Video: More Vectorization Examples<br>
		○ Neural network programming guideline<br>
			§ Whenever possible, avoid explicit for-loops.<br>
				□ Bad example<br>
					® u = Avui = ∑(Ai * v)u = np.zeros((n, 1))for i …    for j …        u[i] += A[i][j] * v[j]<br>
				□ Good example<br>
					® u = np.dot(A, v)<br>
		○ Vectors and matrix valued functions<br>
			§ Say you need to apply the exponential operation on every element of a matrix/vector.<br>
				□ u = np.zeros((n, 1))for i in range(n):    u[i] = math.exp(v[i])<br>
				□ Calculation examples<br>
					® import numpy as np<br>
					® u = np.exp(v)<br>
					® np.log(v)<br>
					® np.abs(v)<br>
					® np.maximum(v, 0)<br>
					® v\**2<br>
		○ Logistic regression derivatives<br>
			§ dw = np.zeros((n_x, 1))<br>
			§ dw += x(i)dz(i)<br>
			§ dw /= m<br>
	- Video: Vectorizing Logistic Regression<br>
		○ Vectorizing Logistic Regression<br>
			§ First set<br>
				□ z(1) = wTx(1) + b<br>
				□ a(1) = sigmoid(z(1))<br>
			§ Second set<br>
				□ z(2) = wTx(2) + b<br>
				□ a(1) = sigmoid(z(2))<br>
			§ Third set<br>
				□ z(3) = wTx(3) + b<br>
				□ a(1) = sigmoid(z(3))<br>
			§ Vectorized calculation<br>
				□ z = np.dot(w.T, x) + b<br>
					® "broadcasting"<br>
					® [z(1), z(2), ..., z(m)] = wTx + [b(1), b(2), ..., b(m)]<br>
				□ A = [a(1), a(2), ..., a(m)] = sigmoid(z)<br>
	- Video: Vectorizing Logistic Regression's Gradient Output(Computation)<br>
		○ Vectorizing Logistic Regression<br>
			§ (Check slide)<br>
		○ Implementing Logistic Regression<br>
			§ for iter in range(1000):<br>
				□ z = np.dot(w.T, X) + b<br>
				□ A = sigmoid(z)<br>
				□ dZ = A - Y<br>
				□ dw = 1/m * X * dZT<br>
				□ db = 1/m * np.sum(dZ)<br>
				□ w := w - alpha * dw<br>
				□ b := b - alpha * db<br>
	- Video: Broadcasting in Python<br>
		○ Broadcasting example<br>
			§ Cal = A.sum(axis=0)<br>
				□ [59.   239.    155.4    76.9]<br>
			§ 100 * A/Cal.reshape(1, 4)<br>
				□ [[94.91525424    0.          …    ] [   2.03389831    43.51…       ] [ … ]]<br>
				□ A / Cal<br>
					® (3, 4) / (1, 4)<br>
		○ General Principle<br>
			§ (m, n) & [+, -, *, /]<br>
				□ (1, n) → (m, n)<br>
				□ (m, 1) → (m, n)<br>
			§ Matlab / Octave: bsxfun<br>
	- Video: A note on python/numpy vectors<br>
		○ Python Demo<br>
		○ Python / numpy vectors<br>
			§ a = np.random.randn(5)<br>
				□ a.shape = (5, )<br>
				□ "rank 1 array"<br>
				□ DON'T USE!<br>
			§ a = np.random.randn(5, 1)<br>
				□ a.shape = (5, 1)<br>
				□ Column vector<br>
			§ a = np.random.randn(1, 5)<br>
				□ a.shape = (1, 5)<br>
				□ Row vector<br>
			§ assert(a.shape == (5, 1))<br>
				□ a = a.shape((5,1))<br>
	- Video: Quick tour of Jupyter/iPython Notebooks<br>
	- Video: Explanation of logistic regression cost function (optional)<br>
		○ Logistic regression cost function<br>
			§ If y = 1: p(y|x) = y_hat<br>
			§ If y = 0: p(y|x) = 1 - y_hat<br>
			§ To sum up, it's p(y|x)<br>
			§ p(y|x) = y_haty (1 - y_hat)(1 - y)<br>
				□ If y = 1: p(y|x) = y_hat (1 - y_hat)0<br>
				□ If y = 0: p(y|x) = y_hat0 (1 - y_hat)1=  1 x (1 - y_hat) = 1 - y_hat<br>
				□ Log p(y|x) = log(y_hat) (1 - y_hat)(1-y)= y log(y_hat) + (1 - y) log(1 - y_hat)= - L(y_hat, y)<br>
		○ Cost on m examples<br>
			§ log p(labels in training set) = log∏1~mp(y(i)|x(i))<br>
			§ log p(…) = ∑1~m log p(y(i)|x(i))(log p(y(i)|x(i)) = -L(y_hat(i)|y(i)) )= -∑1~mL(y_hat(i)|y(i))<br>
			§ Cost (to minimize)<br>
				□ J(w, b) = 1/m ∑1~mL(y_hat(i)|y(i))<br>
			§ "Maximum likelihood estimation"<br>
	- Quiz: Neural Network Basics<br>
	- Reading: Deep Learning Honor Code<br>
	- Reading: Programming Assignment FAQ<br>
	- Practice Programming Assignment: Python Basics with numpy (optional)<br>
	- Programming Assignment: Logistic Regression with a Neural Network mindset<br>
	- Video: Pieter Abbeel Interview<br>
