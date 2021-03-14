# Machine Learning

### Essence of ML
<br>

<details close>
    <summary>What is machine learning (ML)?</summary>

As the name implies, we allow the machine (or the computer) to learn. **Learn from whom?** The machine learns from a large amount of data provided by us. **What is the goal of learning?** The machine learns some patterns or rules from the data. And in return, it gives some insights into additional data in the same form. In other words, the machine tries to propose a hypothesis, which describes the behavior of the entire dataset. When adding new cases to the dataset, the machine can predict the behavior of these cases using the hypothesis. **How do we know if the hypothesis is valid?** A quantitative method is needed to assess the performance of the hypothesis. Intuitively, we can quantify the difference between the predicted behavior and the true behavior. This is known as the <span style="color:red">*Cost Function*</span>. **How do we optimize the hypothesis if the performance is poor?** We systematically update the hypothesis to minimize the cost function using a mathematical approach called <span style="color:red">*Gradient Descent*</span>.

__A formal definition of ML__: a computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

ML includes a wide range of topics, 
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110216068-b1d79800-7e72-11eb-8476-f5b26388aba7.png">
</p>
</details>
<br>

<details close>
    <summary>Supervised learning vs. Unsupervised learning</summary>

**Supervised learning** refers to the task of learning a hypothesis that predicts an output behavior with a given input case, based on example input-output pairs. The machine is supervised by us (or the expected output), and the goal is to minimize the cost function. **Unsupervised learning** refers to the process of learning patterns from untagged data. Since there is no guidance from the expected output, the machine has no "supervisor", and there is no cost function to minimize. Unsupurvised models are typically evaluated with convergence tests.
</details>
<br>

<details close>
    <summary>Gradient descent</summary>
    
Imagine that the mathematical expression of our hypothesis f(x) contains a coefficient. Each time we change the value of the coefficient, we get a new hypothesis. For any given hypothesis, there is a corresponding cost value. If we plot the cost as a function of the coefficients, we get a curve. Our goal is to find the value of the coefficients that minimize the cost of the hypothesis. 
**Gradient descent** is the mathematical process of finding the optimized coefficient values. Looking at the figure blow: we start at a random position on the hillside, and we want to get to the bottom of the valley. The slope(gradient) tells us the correct direction toward the bottom. In this case, the slope is negative and we have to move to the right; if we start from the opposite side of the mountain, the slope will be positive and we have to move to the left. As we move down the mountain, the magnitude of the slope becomes smaller, and we take smaller steps (if the steps are too large, we will walk passby the valley).
    
<br>
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110220826-e016a100-7e8d-11eb-90ee-5c7cda936b19.png">
</p>    
</details>
<br>

<details close>
    <summary>Training, validation, and test sets</summary>
    
**Training set** refers to the data provided during optimization; while **Validation set** refers to new data that the model have not seen during the training, it's often used for hyper-parameter tuning (learning rate, regularization coefficient, etc.). There is often another type of data called **Test set**, which is solely used for reporting model performance after the model has been locked.
<br>
</details>
<br>

<details close>
    <summary>Overfitting & Underfitting</summary>

Overfitting or underfitting is a term commonly used to describe the performance of a given model on the validation set compared to the training set. When the model performs equally poorly on training set and validation set, the model has a high **bias** and it is **underfitting.** When the model performs well on the training set, but poorly on the validation set, the model has a high <span style="color:red">*variance*</span> and it is **overfitting.**
A more complicated model is needed if in the case of underfitting; while a simpler model or regularization is needed in the case of overfitting.
<br>
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110220463-8319eb80-7e8b-11eb-8199-3bf85c54a0e5.png">
</p>
</details>
<br>

<details close>
    <summary>Regularization</summary>
<br>
Regularization generally refers to the process of suppressing overfitting. There are several common ways to implement regularization, namely L0, L1, L2, early stopping, and dropout for neural networks. L1 regularization and L2 regularization are the most commonly used ones.
    
Using the linear regression with L1, L2 regularization as an example. Look at the figure below, the red dot is the global minimum for the cost function represented by the elliptical contours without regularization. The model corresponding to this global minimum is overfitted, so we have to move to an adjacent point where it also satisfies the constraint on the weight represented by the square or circle. 
    
L1 regularization:
     |w<sub>1</sub>| + |w<sub>2</sub>| < constraint 
    
L2 regularization:
     |w<sub>1</sub>|<sup>2</sup> + |w<sub>2</sub>|<sup>2</sup> < constraint 
 
Contour lines have a much greater chance of intersecting a square on the axis than a circle. This means L1 regularization tends to completely remove some features (feature selection) by setting w to 0, L2 rarely remove features.
    
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110222803-8d8fb180-7e9a-11eb-85bc-b2faa21c9a4d.png">
</p>
</details>

### List of Content 
- [Linear Regression](#Linear-Regression-from-Scratch)
- [Logistic Regression](#Logistic-Regression-from-Scratch)
- [Support Vector Machine](#Support-Vector-Machine)
- [K- Nearest Neighbor](#K-Nearest-Neighbors)
- [Naive Bayes](#Naive-Bayes)
- [Decision Tree & Random Forest](#Decision-Tree-&-Random-Forest)



# Linear Regression from Scratch

Linear regression is one the simplest ML models, yet it is also a very powerful model for regression tasks. Let's see how to build a linear regression model from scratch.

### 1 Independent variable --> 1 Dependent variable

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Creating training data with noise
X = np.linspace(1,100,1000) + np.random.randn(1000)
y = np.linspace(1,100,1000) * 6 + 3  + 25*np.random.randn(1000)

plt.plot(X,y,'b.')
plt.xlabel("Input X")
plt.ylabel('Target y')
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110224502-df880580-7ea1-11eb-8c1d-80c3ca5502dd.png">
</p>

<details close>
    <summary>Mathematical derivation</summary>
    
<img width="600" alt="image" src="https://user-images.githubusercontent.com/66216181/110228014-4669e680-7ec3-11eb-8f7c-bdf3d16519ab.png">

    
</details>

```python
# Propose a model ==> (w,b)
np.random.seed(42)
w = np.random.randn(1)
b = np.random.randn(1)

# Get output using the proposed model ==> ŷ 
y_hat = X * w + b 

# Evaluate performance of the initial model ==> MSE
mse = sum((y_hat - y)**2)/len(X)
print('MSE of initial model: {}'.format(mse))

#Gradient descent, update (w,b)
w = w - 0.00005 * (-2/len(X) * sum((y - y_hat) * X))
b = b - 0.002 * (-2/len(X) * sum(y - y_hat))

# Re-evaluation after 1-step gradient descent
y_hat = X * w + b 
mse = sum((y_hat - y)**2)/len(X)
print('MSE of 1-step model: {}'.format(mse))

#How about 20 steps:
loss = [105992.64543369334,69772.88213923965]
for i in range(18):
    w = w - 0.00005 * (-2/len(X) * sum((y - y_hat) * X))#5e-5 is the learning rate for w
    b = b - 0.002 * (-2/len(X) * sum(y - y_hat))#2e-3 is the learning rate for b
    y_hat = X * w + b 
    mse = sum((y_hat - y)**2)/len(X)
    loss.append(mse)
    print('MSE of {}-step model: {}'.format(i+2,mse))
    
print('Final model: w = {}, b = {}'.format(w[0],b[0]))
```
```
MSE of initial model: 105320.37198727363
MSE of 1-step model: 46254.72057558888
MSE of 2-step model: 20523.186963247765
MSE of 3-step model: 9313.425003791546
MSE of 4-step model: 4429.969721603055
MSE of 5-step model: 2302.5250119991715
MSE of 6-step model: 1375.7167365904795
MSE of 7-step model: 971.957199719738
MSE of 8-step model: 796.060134635507
MSE of 9-step model: 719.4297409121444
MSE of 10-step model: 686.0441714956168
MSE of 11-step model: 671.4979066293228
MSE of 12-step model: 665.1588549746958
MSE of 13-step model: 662.3952214515762
MSE of 14-step model: 661.1891974250099
MSE of 15-step model: 660.6617388879029
MSE of 16-step model: 660.4298965547194
MSE of 17-step model: 660.3268412816484
MSE of 18-step model: 660.2798951295513
MSE of 19-step model: 660.2573964628607
Final model: w = 6.017417448192188, b = 3.233686587040213
```
```python
plt.plot(np.linspace(0,21,20),loss,'bo-',markersize=5)
plt.xlabel('Number of iterations')
plt.ylabel('MSE')
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110224654-0266e980-7ea3-11eb-8d65-16be016d9b87.png">
</p>

The loss drops monotonically after each iteration, and eventually plateaus to the global minimum. The parameters are optimized from the randomly initialized values. Let's see how the final model predicts the data.

```python
# Final Model:
plt.plot(X,y,'b.')
plt.plot(X,y_hat,'r',linewidth=3)
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110224673-33dfb500-7ea3-11eb-97c7-da1ab45ede57.png">
</p>

### n Independent variable --> m Dependent variable

When the sample space is nolonger 1D, the computing becomes much more challenging since we have to write out all the parameter update steps for every single dimension (w<sub>11</sub>,w<sub>12</sub>,...,w<sub>mn</sub> $). Linear algebra offers a much easier way to carry out the computation. Let's seem how to vactorize the process. 

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
%matplotlib inline
```
```python
#Creating random data

X = 100*np.random.randn(2000).reshape(1000,2) + 25*np.random.rand(1000,2)

y1 =  3 * X[:,0].reshape(-1,1) + 5 * X[:,1].reshape(-1,1) +7 + 5 * np.random.rand(1000,1)
y2 = -2 * X[:,0].reshape(-1,1) + X[:,1].reshape(-1,1) +5 + 5 * np.random.rand(1000,1)

# Visualizing Data in 3D
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = fig.gca(projection='3d')

a = np.linspace(-300, 300, 1000)
b = np.linspace(-300, 300, 1000)

ax.scatter3D(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), y1, color = "red",alpha=0.2)
ax.scatter3D(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), y2, color = "blue",alpha=0.2)

ax.set_title('3D-Visualization')
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110224750-f2033e80-7ea3-11eb-97c3-0727b4ffa2da.png">
</p>

<details close>
    <summary>Mathematical derivation</summary>
 <p align="center">
<img width="600", alt="image" src="https://user-images.githubusercontent.com/66216181/110224758-05aea500-7ea4-11eb-8f74-446d064820c4.png">
</p>   
</details>

```python
#Create tensor
X = torch.from_numpy(X)
Y = torch.from_numpy(np.hstack((y1,y2)))

# Propose a model ==> (w,b)
torch.manual_seed(42)
w = torch.randn(2,2,requires_grad=True, dtype = torch.float64)
b = torch.randn(2,requires_grad=True, dtype = torch.float64)

# Get output using the proposed model ==> ŷ 
Y_hat = X @ w.t() + b

# Evaluate performance of the initial model ==> MSE
mse = torch.sum((Y_hat-Y)**2) / Y.numel()
print('MSE of initial model: {}'.format(mse))

#Gradient descent, update (w,b)
mse.backward()

# Re-evaluation after 1-step gradient descent
with torch.no_grad():
    w -= w.grad * 0.00003
    b -= b.grad * 0.00003
    w.grad.zero_()
    b.grad.zero_()
    
Y_hat = X @ w.t() + b
mse = torch.sum((Y_hat-Y)**2) / Y.numel()
print('MSE of 1-step model: {}'.format(mse))

# How about 10 more steps:
loss = [193148.77403374686,90220.90682016108]
for i in range(10):
    mse.backward()
    with torch.no_grad():
        w -= w.grad * 0.00003
        b -= b.grad * 0.00003
        w.grad.zero_()
        b.grad.zero_()
    Y_hat = X @ w.t() + b
    mse = torch.sum((Y_hat-Y)**2) / Y.numel()
    loss.append(mse)
    print('MSE of {}-step model: {}'.format(i+2,mse))
```
```
MSE of initial model: 193148.77403374686
MSE of 1-step model: 90220.90682016108
MSE of 2-step model: 42242.51982448351
MSE of 3-step model: 19839.987286519317
MSE of 4-step model: 9360.551662814096
MSE of 5-step model: 4448.997980016644
MSE of 6-step model: 2142.2989524659283
MSE of 7-step model: 1056.6131637700028
MSE of 8-step model: 544.4509851013239
MSE of 9-step model: 302.2653797860315
MSE of 10-step model: 187.45734657239717
MSE of 11-step model: 132.89111191947825
```
```python
plt.plot(np.linspace(0,13,12),loss,'bo-',markersize=5)
plt.xlabel('Number of iterations')
plt.ylabel('MSE')
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110224788-4efef480-7ea4-11eb-8d80-3fade808ef12.png">
</p>

```python
print('Final model: w = {}, b = {}'.format(w,b))
```
```
Final model: w = tensor([[ 2.9668,  4.9551],
        [-1.9333,  0.9958]], dtype=torch.float64, requires_grad=True), b = tensor([-1.1076, -0.1861], dtype=torch.float64, requires_grad=True)
```

```python
# Visualizing Data in 3D
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = fig.gca(projection='3d')


a = np.linspace(-300, 300, 1000)
b = np.linspace(-300, 300, 1000)

A, B = np.meshgrid(a, b)
C = 2.9668 * A + 4.9551 * B - 1.1076
D = -1.9333 * A + 0.9958 * B - 0.1861

ax.plot_surface(A, B, C, alpha=0.7,color = "red")
ax.plot_surface(A, B, D, alpha=0.7,color = "blue")
ax.scatter3D(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), y1, color = "red",alpha=0.2)
ax.scatter3D(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), y2, color = "blue",alpha=0.2)


ax.set_title('3D-Visualization')

```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110224815-9be2cb00-7ea4-11eb-8206-96dd896abade.png">
</p>

# Logistic Regression from Scratch

Logistic regression in the simplest classification model, and it is often considered as the building block of neural networks.

```python
from sklearn.datasets import make_classification
data = make_classification(n_samples=300,n_features=1,n_informative=1, n_redundant=0,n_clusters_per_class=1,random_state=4)
X = data[0]
y = data[1].reshape(-1,1)
plt.plot(X,y,'o')
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110228074-d314a480-7ec3-11eb-9e26-c9cf4e5102c9.png">
</p>

<details close>
    <summary>Mathematical derivation</summary>
 <p align="center">
<img width="600", alt="image" src="https://user-images.githubusercontent.com/66216181/110228287-8fbb3580-7ec5-11eb-91b1-276b8544fac3.png">
</p>  
</details>

```python
# Propose a model ==> (w,b)
np.random.seed(42)
w = np.random.randn(1)
b = np.random.randn(1)

# Get output using the proposed model ==> ŷ 
def sigmoid(input):
    return 1/(1+np.exp(-input))
y_hat = sigmoid(X * w + b)  

# Evaluate performance of the initial model ==> Cross Entropy
cost = - sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))/len(X)
print('Cost of initial model: {}'.format(cost[0]))

#Gradient descent, update (w,b)
w = w - 1 * (-1/len(X) * sum((y - y_hat) * X))
b = b - 1 * (-1/len(X) * sum(y - y_hat))

# Re-evaluation after 1-step gradient descent
y_hat = sigmoid(X * w + b) 
cost = - sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))/len(X)
print('Cost of initial model: {}'.format(cost[0]))

#How about 50 steps:
loss = [0.47441666,0.40120507]
for i in range(48):
    w = w - 1 * (-1/len(X) * sum((y - y_hat) * X))#0.3 is the learning rate for w
    b = b - 1 * (-1/len(X) * sum(y - y_hat))
    y_hat = sigmoid(X * w + b) 
    cost = - sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))/len(X)
    loss.append(cost)
    if (i + 2) % 10 == 0:
        print('Cost of {}-step model: {}'.format(i+2,cost[0]))
    
    
print('Final model: w = {}, b = {}'.format(w[0],b[0]))
```

```
Cost of initial model: 0.4758919497866085
Cost of initial model: 0.35993527399552555
Cost of 10-step model: 0.14083923937680098
Cost of 20-step model: 0.0974768205256391
Cost of 30-step model: 0.07903043468721682
Cost of 40-step model: 0.06850834864785582
Final model: w = 3.8450567371871887, b = 0.3648425287090521
```
As expected, the cost drops monotonically.

```python
plt.plot(np.arange(50),loss,'bo-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
```
 <p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110228194-da887d80-7ec4-11eb-8555-f88623cb1299.png">
</p> 

Let's see what the sigmoid function look like.
```python
plt.plot(X,y,'o')
plt.plot(X,y_hat,'.')
```
 <p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110228226-102d6680-7ec5-11eb-98c3-e87cf269d0ac.png">
</p> 

# Support Vector Machine

### Classification

Support vector machine classifier (SVC) is a widely used algorithm often for classification tasks. An SVM maps training examples to points in space, and try to find two parallel hyperplanes to separate the points from different categories while maximize the distance between the two hyperplanes. The region bounded by these two hyperplanes is called the *margin*, and the points lie on the margin is called the *support vectors.* When the points are not linearly separable in the current sample space, SVM use a *kernel trick* to map points into higher dimensional spaces, and systematically search for hyperplances that can separate the points. The hyperplanes can be defined by their normal vector (w), and their bias (b).

<p align="center">
<img width="360", alt="image" src="https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png">
</p>

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y =  make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1, random_state=9)
x1 = X[:,0]
x2 = X[:,1]
label = y.copy()
y[y == 0]= -1 #prepare lable for SVM
sns.scatterplot(x1,x2,hue=label,palette='Set2',)
plt.xlabel('x1')
plt.ylabel('x2')
y = y.reshape(-1,1)
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110886866-ee5e2600-82ae-11eb-9732-199cc19d409f.png">
</p>

<details close>
    <summary>Mathematical derivation</summary>
    
#### Conceptually 
We calculate the distance of each point to the hyperplane which lies in the middle of the margin. Mathematically this can be done by projecting the vector representing each point onto the normal vector of the hyperplane. If the distance is larger than some value, we say the point is classified correctly. If the distance is smaller, or even on the wrong side of the hyperplane, we say the point is classified incorrectly. *Hinge loss* is used to evaluate the performance of the hyperplane. The hinge score is associated with the calculated distance and the corresponding label of the point. If the point is on the correct side of the hyperplane but lie on the margin, then it has a loss between 0 to 1. If the point is on the wrong side, it has a loss greater than 1.
    
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110412579-07b66680-8052-11eb-9491-d4714677b6cf.png">
</p>
    

#### Math Implementation
    
The label y is converted to ± 1.
    
Hinge score:  y<sub>i</sub> * (w·X<sub>i</sub> - b)
    
*If the point is on the positive side and it's correctly labeled as +1, then the hinge score will be positive; if the point in on the positive side but it has a -1 label, then the hinge score will be negative. Similarly, if the point is on the negative side and it's correctly labeled as -1, the hinge score is positive (the projection w·X is a negative value); if the pint has a +1 label, then the hinge score is negative.*

Hinge loss:  max(0, 1 - y<sub>i</sub> * (w·X<sub>i</sub> - b))


Cost function:
    
<img width="530" alt="image" src="https://user-images.githubusercontent.com/66216181/110887196-78a68a00-82af-11eb-9f8c-709c3f33c62d.png">
    
</details>

```python
# Propose a model ==> (w,b)

w=np.zeros(2).reshape(2,1)
b=np.zeros(1).reshape(1,1)
# Get output using the proposed model ==> ŷ 
def cal_score(point_v,lable):
    return lable * (X @ w - b)
s = cal_score(X,y)

# Evaluate performance of the initial model ==> Hinge Loss
def cal_hinge(score):
    hinge_loss = 1 - score
    hinge_loss[hinge_loss < 0] = 0 #
    cost = 0.5* sum(w**2)  + sum(hinge_loss)/len(y)
    return hinge_loss, cost

_, J = cal_hinge(s)
loss = [J[0]]
print('Cost of initial model: {}'.format(J[0]))

#Batch Gradient descent, update (w,b)
def cal_grad(point_v,lable,lambda_):
    hinge, _ = cal_hinge(cal_score(point_v,lable))
    grad_w = np.zeros(w.shape)
    grad_b = np.zeros(b.shape)
    for i, h in enumerate(hinge):
        if h == 0:
            grad_w += lambda_ * w
        else:
            grad_w += lambda_ * w - (X[i] * y[i]).reshape(-1,1)
            grad_b += y[i]
            
    return grad_w/len(X), grad_b/len(X)

grad_w,grad_b = cal_grad(X,y,0.02)
w = w - 0.001*grad_w
b = b - 0.001*grad_b

# Re-evaluation after 1-step gradient descent
s = cal_score(X,y)
_, J = cal_hinge(s)
print('Cost of 1-step model: {}'.format(J[0]))
loss.append(J[0])
#How about 30 steps:

for i in range(198):
    grad_w,grad_b = cal_grad(X,y,0.02)
    w = w - 0.001*grad_w
    b = b - 0.001*grad_b
    s = cal_score(X,y)
    _, J = cal_hinge(s)
    loss.append(J[0])
    if (i+2)%25 == 0:
        print('Cost of {}-step model: {}'.format(i+2,J[0]))
    
    
print('Final model: w = {}, b = {}'.format(w,b))
```

```
Cost of initial model: 1.0
Cost of 1-step model: 0.9641497982696956
Cost of 25-step model: 0.2194894228521624
Cost of 50-step model: 0.05047965421757659
Cost of 75-step model: 0.03799414902875944
Cost of 100-step model: 0.03459882628099443
Cost of 125-step model: 0.033421248550687954
Cost of 150-step model: 0.033433728271506535
Cost of 175-step model: 0.033513028899963174
Final model: w = [[ 0.14492139]
 [-0.19914711]], b = [[-0.01344]]
```

```python
plt.plot(np.arange(0,200),loss,'bo')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110887336-bc00f880-82af-11eb-8101-67e62c02ff6b.png">
</p>
    

```python
sns.scatterplot(x1,x2,hue=label,palette='Set2',)
plt.xlabel('x1')
plt.ylabel('x2')

plt.plot(x1,-(w[0]*x1-b[0])/w[1])
plt.plot(x1,-(w[0]*x1-b[0]+1)/w[1],'y-.')
plt.plot(x1,-(w[0]*x1-b[0]-1)/w[1],'y-.')
plt.ylim(-12,4)
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110887404-dfc43e80-82af-11eb-86ad-c7f3c9657ed2.png">
</p>


### Regression

Support vector machine regressor (SVR) is similar to SVC. An SVR also maps training examples to points in space, and try to find two parallel hyperplanes to capture all the points. 

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110880789-c7025b80-82a4-11eb-8e80-da8447b14a31.png">
</p>


```python
import matplotlib.pyplot as plt
import numpy as np

# Creating training data with noise
X = np.linspace(1,10,1000) + 0.2*np.random.randn(1000)
y = 2 * X + 3 + 0.2*np.random.randn(1000)
```

<details close>
    <summary>Mathematical derivation</summary>
    
#### Conceptually 
We define a hyperplane with weight and bias. Moving the hyperplane up and down by a distance of epsilon will create a margin. We try to capture as many data points as possible with the margin. If the point is within the margin, there is no loss, otherwise, the loss is the distance from the point to the margin. 
    

<img width="602" alt="image" src="https://user-images.githubusercontent.com/66216181/110887561-2b76e800-82b0-11eb-8136-93e38ba38152.png">

</details>

```python
# Parameter initialization
epsilon = 0.5
lambda_ = 0.001

w = np.zeros(1)
b = np.zeros(1)

# Calculating the loss
def cal_loss(w, b, X, y, epsilon, lambda_):
    y_hat = w * X + b
    loss = np.abs(y - y_hat) - epsilon
    loss[loss < 0] = 0
    return loss + lambda_ * w**2

#Batch Gradient descent, update (w,b)
def grad(w, b, X, y, epsilon, lambda_):
    loss = cal_loss(w, b, X, y, epsilon, lambda_)
    
    grad_w = 0
    grad_b = 0
    
    for i, loss_i in enumerate(loss):
        if np.abs(y[i] - w * X[i] - b) - epsilon <= 0:
            grad_w += lambda_ * w
        else:
            if y[i] >= w * X[i] + b:
                grad_w += lambda_ * w - X[i]
                grad_b += -1
            else:
                grad_w += lambda_ * w + X[i]
                grad_b += 1
    w -= 0.006 * grad_w/len(X)
    b -= 0.05 * grad_b/len(X)
    return w,b

loss_history=[]
for i in range(500):
    loss_history.append(sum(cal_loss(w, b, X, y, epsilon, lambda_)))
    w,b = grad(w, b, X, y, epsilon, lambda_)
print(w,b)
```

```
[1.99554697] [3.03405]

```

```python
plt.plot(np.arange(500),loss_history,'o')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110887725-68db7580-82b0-11eb-8192-f1df4e7dec06.png">
</p>

```python
plt.plot(X,y,'yo')
plt.plot(X,w*X+b-epsilon ,'b-')
plt.plot(X,w*X+b+epsilon ,'b-')
plt.xlim(3,5)
plt.ylim(7,15)
```


<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/110887770-7bee4580-82b0-11eb-8e5b-336a15289f55.png">
</p>

# K-Nearest Neighbors 

K-Nearest Neighbors (KNN) is a very representative instance-based supervised learning algorithm, where new instances are predicted based on their similarity or "distance" from the training instances. Unlike the model based machine learning algorithms, it does not optimize model parameters through gradient descent.

### Classification
For classification, the new instance is labeled as the most frequent class in its kNN, like a voting process.

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y =  make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.5, random_state=42)

x1 = X[:,0]
x2 = X[:,1]
sns.scatterplot(x1,x2,hue=y,palette='Set2',)
plt.xlabel('x1')
plt.ylabel('x2')
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/111084974-277edc00-84e3-11eb-886d-7a3da581f743.png">
</p>

```python
#Set parameters
dis_power = 2
num_neighbors = 5

# Calculating the distance between new intstance and training instance
def cal_dis(x_new, x_train, power = dis_power):
    dis = 0
    for dim in range(len(x_train)):
        dis += abs(x_train[dim]-x_new[dim])**power
    return dis**(1/power)

# Find K nearest neighbors
def get_knn(x_new, X, power = dis_power, k = num_neighbors):
    dis = []

    for i in range(len(X)):
        dis.append(cal_dis(x_new, X[i], power))

    dis_sort = pd.DataFrame(dis,columns=['distance'])
    dis_sort.sort_values('distance',inplace=True)
    return list(dis_sort.head(k).index)
    
# Calculating the target value using KNN.
def pred(x_new, X, power = dis_power, k = num_neighbors):
    y_pred =[]
    for i in x_new:
        y_knn = []
        for j in get_knn(i, X, power, k):
            y_knn.append(y[j])
        y_pred.append(max(set(y_knn), key=y_knn.count))
    return y_pred
```

```python
a = np.arange(-13, 8, 0.1)
b = np.arange(-10, 11, 0.1)
aa, bb = np.meshgrid(a, b)
abpairs = np.dstack([aa, bb]).reshape(-1, 2)

clf = pred(abpairs, X, power = dis_power, k = num_neighbors)

h = plt.contourf(a,b,np.array(clf).reshape(210,210))
sns.scatterplot(x1,x2,hue=y,palette='Set2',)
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim(-10,11)
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/111085010-4f6e3f80-84e3-11eb-940f-e78f9c0ee34a.png">
</p>

### Regression
For regression, the target value of the new instance is the mean of all target values of its KNN.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Creating training data with noise
np.random.seed(42)
X = np.linspace(1,10,100) + 2*np.random.randn(100)
y = 0.2 * X**3 + X**2 -4*X -7 + 5*np.random.randn(100)
plt.plot(X,y,'bo')
plt.xlabel('X')
plt.ylabel('y')
X = X.reshape(-1,1)
y = y.reshape(-1,1)
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/111085039-847a9200-84e3-11eb-8c0c-0941f9cc1a7e.png">
</p>

```python
#Set parameters
dis_power = 2
num_neighbors = 5

# Calculating the distance between new intstance and training instance
def cal_dis(x_new, x_train, power = dis_power):
    dis = 0
    for dim in range(len(x_train)):
        dis += abs(x_train[dim]-x_new[dim])**power
    return dis**(1/power)

# Find K nearest neighbors
def get_knn(x_new, X, power = dis_power, k = num_neighbors):
    dis = []

    for i in range(len(X)):
        dis.append(cal_dis(x_new, X[i], power))

    dis_sort = pd.DataFrame(dis,columns=['distance'])
    dis_sort.sort_values('distance',inplace=True)
    return list(dis_sort.head(k).index)
    
# Calculating the target value using KNN.
def pred(x_new, X, power = dis_power, k = num_neighbors):
    y_pred =[]
    for i in x_new:
        y_knn = []
        for j in get_knn(i, X, power, k):
            y_knn.append(y[j])
        y_pred.append(np.mean(y_knn))
    return y_pred
```

```python
X = np.linspace(1,10,100) + 2*np.random.randn(100)
y = 0.2 * X**3 + X**2 -4*X -7 + 5*np.random.randn(100)
plt.plot(X,y,'bo')
plt.xlabel('X')
plt.ylabel('y')
X = X.reshape(-1,1)
y = y.reshape(-1,1)
plt.plot(np.linspace(0,15,100).reshape(-1,1),
         pred(np.linspace(0,15,100).reshape(-1,1),X,2,5),'r-',linewidth=2.5)
```
<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/111085070-aaa03200-84e3-11eb-9333-a7a37151fe39.png">
</p>

# Naive Bayes
Naive Bayes is a conditional probability model mostly used for classification problem. 

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y =  make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.5, random_state=42)

x1 = X[:,0]
x2 = X[:,1]
sns.scatterplot(x1,x2,hue=y,palette='Set2',)
plt.xlabel('x1')
plt.ylabel('x2')
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/111084974-277edc00-84e3-11eb-886d-7a3da581f743.png">
</p>

<details close>
    <summary>Mathematical derivation</summary>
    
<p align="center">
<img width="1204", alt="image" src="https://user-images.githubusercontent.com/66216181/111085123-f2bf5480-84e3-11eb-80ac-9b0e31ec3f0d.png">
</p>   

</details>

```python
# Training process, calculate probability density function for each feature in each class.
def cal_stat(X,y):
    all_class = np.unique(y)
    n_class = len(all_class)
    n_sample, n_feat = X.shape
    stat_mean = np.zeros((n_class, n_feat))
    stat_var = np.zeros((n_class, n_feat))
    priors = np.zeros(n_class)

    for i, c in enumerate(all_class):
        stat_mean[i,:] = X[np.where(y==c)].mean(axis=0)
        stat_var[i,:] = X[np.where(y==c)].var(axis=0)
    return stat_mean, stat_var

# Use the PDF to find the class to which the new instance most likely belongs.
def NaiveBayes(x_new, X, y):
    stat_mean, stat_var = cal_stat(X,y)
    log_likelihood = np.zeros((len(np.unique(y)), X.shape[1]))
    log_prior = np.log((np.bincount(y)[np.bincount(y)!=0]/len(y))).reshape(-1,1)

    res = []

    for x_i in x_new:
        for i, c in enumerate(np.unique(y)):
            log_likelihood[i,:] = np.log(np.exp(- (x_i - stat_mean[i,:])**2 / 
                                                (2 * stat_var[i,:])) / np.sqrt(2 * np.pi * stat_var[i,:]))
        posterior = log_prior + log_likelihood.sum(axis=1).reshape(-1,1)
        res. append(np.unique(y)[np.argmax(posterior)])
    return res

```

```python
a = np.arange(-13, 8, 0.1)
b = np.arange(-10, 11, 0.1)
aa, bb = np.meshgrid(a, b)
abpairs = np.dstack([aa, bb]).reshape(-1, 2)
clf = NaiveBayes(abpairs, X, y)

h = plt.contourf(a,b,np.array(clf).reshape(210,210))
sns.scatterplot(x1,x2,hue=y,palette='Set2',)
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim(-10,11)
```

<p align="center">
<img width="360", alt="image" src="https://user-images.githubusercontent.com/66216181/111085164-2b5f2e00-84e4-11eb-8b2b-4722930d760d.png">
</p>   
