# deep-learning-challenge

Report on the Performance of the Deep Learning Model for Alphabet Soup

1. Overview of the Analysis:
The purpose of this analysis is to create a deep learning model using TensorFlow and Keras to predict the success of Alphabet Soupfunded organizations based on certain features provided in the dataset. The analysis involves preprocessing the data, compiling, training, and evaluating the model, and optimizing it to achieve a target predictive accuracy higher than 75%.

2. Results:
Data Preprocessing:
Target Variable(s): 
The target variable for the model is "IS_SUCCESSFUL," which indicates whether the organization was successful or not.

<img width="352" alt="image" src="https://github.com/techgirlnerd/deep-learning-challenge/assets/145892936/26f43f2e-5d02-4e16-8434-0041c82b48c3">


Feature Variable(s): 
The feature variables for the model include all columns except "EIN," "NAME," and "IS_SUCCESSFUL."
 <img width="468" alt="image" src="https://github.com/techgirlnerd/deep-learning-challenge/assets/145892936/1ec42636-e45a-4682-a618-2dc50ecb5ecf">

Removed Variables: 
The columns "EIN" and "NAME" were removed from the input data because they are neither targets nor features.
<img width="111" alt="image" src="https://github.com/techgirlnerd/deep-learning-challenge/assets/145892936/38ce2d10-0692-4ab7-8830-510146363cca">

 

Compiling, Training, and Evaluating the Model:
 Neurons, Layers, and Activation Functions:
 The original model consists of one hidden layer with 64 neurons and uses the ReLU activation function. The output layer uses the sigmoid activation function.
 In attempts to optimize the model, different strategies were tested, most prominently:
1.	Strategy 1: Basic Neural Network
•	This strategy involves building a basic neural network with one hidden layer consisting of 64 neurons and ReLU activation function.
•	The model was trained for 30 epochs.
2.	Strategy 2: Increased Neurons and Activation Function
•	In this strategy, the number of neurons in the hidden layers was increased, and the activation function was changed to tanh.
•	It still uses a single hidden layer but with more neurons (256, 128, 64) and different activation functions (ReLU, ReLU, tanh).
•	The model was trained for 50 epochs.
3.	Strategy 3: Dropout and Batch Normalization
•	Strategy 3 incorporates dropout and batch normalization techniques to prevent overfitting and improve generalization performance.
•	Dropping additional columns namely “SPECIAL CONSIDERATIONS”
•	It includes three hidden layers with dropout and batch normalization applied after each dense layer.
•	The model architecture consists of layers with 256, 128, and 64 neurons, respectively, with ReLU activation functions.
•	The model is trained for 50 epochs.

 Target Model Performance:
 The target model performance was to achieve an accuracy higher than 75%. However, the original model and the attempted optimized versions did not meet this target, with accuracies ranging from approximately 73% to 74%.
 Steps I took to Increase Model Performance:
Strategies employed to improve model performance included :
1.	Adjusting input data (dropping columns, binning rare categorical variables)
2.	Increasing the number of neurons and layers
3.	Using different activation functions
4.	Training the model for different number of epochs.

3. Summary:
The overall results indicate that while efforts were made to optimize the deep learning model, the target predictive accuracy of 75% was not achieved. Despite experimenting with dozens of varirants as per my knowledge and online research work, including different architectures and preprocessing techniques, the model's accuracy remained below the desired threshold.

My Recommendations/How to Improve from here:
After going through the intense steps of trying to get the model accuracy past 75% followed by online research work, here are my three recommendations for using different models to solve the classification problem:
1. Support Vector Machines (SVM):
SVM is a powerful supervised learning algorithm used for classification and regression tasks.
Working: 
It works by finding the hyperplane that best separates the classes in the feature space. SVM can handle highdimensional data.
How to Use: After preprocessing the dataset and scaling the features, SVM can be trained using the training data. The choice of kernel (linear, polynomial, radial basis function) and regularization parameter (C) can significantly impact model performance and should be tuned accordingly. Crossvalidation techniques can be employed to get optimum hyperparameters.
Why Use: 
1.	SVM is effective in capturing complex relationships in data and is robust against overfitting, especially in highdimensional feature spaces. 
2.	It works well with both linearly separable and nonlinearly separable data, making it a versatile choice for classification tasks.


2. Random Forest:
Random Forest is an ensemble learning algorithm that constructs more than one decision trees during training and outputs the mode or the mean prediction of the individual trees.
How to Use: 
After preprocessing the dataset and splitting it into training and testing sets, the Random Forest algorithm can be applied directly to the data. The number of trees in the forest and other hyperparameters can be adjusted to optimize model performance. Crossvalidation techniques can be employed to get optimum hyperparameters.
Why Use: 
1.	Random Forest is suitable for classification tasks with categorical and numerical features, as it can handle both types of data effectively. 
2.	It is less prone to overfitting compared to deep learning models and can provide good performance with minimal hyperparameter tuning.

3. XGBoost (Extreme Gradient Boosting):
XGBoost is yet another powerful gradient boosting algorithm known for its efficiency and performance in classification tasks. 
How to Use:
 The dataset can be preprocessed similarly to the neural network approach. After encoding categorical variables and splitting the data, the XGBoost algorithm can be trained and evaluated using the training and testing datasets. Hyperparameters such as learning rate, tree depth, and regularization parameters can be tuned using techniques like grid search or random search to optimize model performance.
Why Use: 
1.	XGBoost often outperforms traditional machine learning algorithms like Random Forest and Gradient Boosting due to its parallelized and optimized implementation.
2.	It can handle large datasets with highdimensional feature spaces efficiently and is robust against overfitting, making it a good choice for solving classification problems.



By exploring these alternative models, we can leverage their unique strengths and capabilities to potentially achieve better performance in predicting the success of charitable organizations funded by Alphabet Soup.

