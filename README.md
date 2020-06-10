# Project Author
SHAKIB BADARPURA

Contact me via- 

mail: shakibb35@gmail.com

linkedIn- https://www.linkedin.com/in/shakib-badarpura-324b2919a/

Phone- +91989282306


# Rainfall-Prediction-using-Neural-Networks
A machine Learning based Multiple linear regression model to predict the rainfall on the basis of different input parameters. The input features includes pressure, temperature, humidity etc.  

# Dataset Used
The dataset used is downloaded from Kaggle and is freely available. The dataset is named as "Austin weather dataset". The dataset is uploaded with the files.

# Methodology
Artificial Neural Networks is one of the most popular machine learning and deep
learning algorithms. They are inspired by human neurons which are capable of making
human like decisions with help of computations. In a Neural Network Architecture there
are three types of layers:

1. Input Layers: It’s the layer in which we give input to our model. The number of
neurons in this layer is equal to total number of features in our data.

2. Hidden Layer: The input from Input layer is then feed into the hidden layer. There can
be many hidden layers depending upon our model and data size. Each hidden layer can
have different numbers of neurons which are generally greater than the number of
features. The output from each layer is computed by matrix multiplication of output of
the previous layer with learnable weights of that layer and then by addition of learnable
biases followed by activation function which makes the network nonlinear.

3. Output Layer: The output from the hidden layer is then fed into an Activation
function like sigmoid, SoftMax or ReLU which then gives the final output.
The data is then fed into the model and output from each layer is obtained this step is
called feedforward, we then calculate the error using an error function, some common
error functions are cross entropy, square loss error etc.
For example, in our case we trained the Neural Networks with different features like
humidity, temperature, pressure etc. and they learn to identify and analyze the rainfall
based on these features using the results of training dataset. The very simple neural
network might contain only one input neuron, one hidden neuron, and one output neuron.
It takes several dependent variables = input parameters, multiplies them by their
coefficients = weights, and runs them through a ReLU activation function and a unit step
function
