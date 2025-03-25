# Machine Learning Projects Portfolio

Welcome to my Machine Learning Projects Portfolio! This repository serves as a central hub linking to my individual machine learning project repositories. Each project demonstrates various machine learning techniques, including data preprocessing, feature engineering, model selection, and evaluation.

## Projects Overview

### [1. Titanic Survival Prediction](https://github.com/sashwattanay/ML-Kaggle-Titanic-Challenge)
- **Description:** Predict survival outcomes of Titanic passengers using Logistic Regression.
- **Key Highlights:**
  - Handled missing data in `Age`, `Embarked`, and `Fare`.
  - Engineered features like `FamilySize` and `IsAlone`.
  - Achieved a Kaggle public leaderboard score of **0.75837**.
- **Skills Demonstrated:** Data cleaning, feature engineering, logistic regression.

### [2. MNIST Digit Classification](https://github.com/sashwattanay/ML-MNIST-project)
- **Description:** Classify handwritten digits from the MNIST dataset using K-Nearest Neighbors (KNN).
- **Key Highlights:**
  - Normalized pixel intensity values using Min-Max Scaling.
  - Removed low-variance features to reduce dimensionality.
  - Achieved a test accuracy of **97.31%** with high precision and recall.
- **Skills Demonstrated:** Feature selection, hyperparameter tuning, cross-validation.

### [3. Classifying Iris Flowers Using Batch Gradient Descent and Early Stopping](https://github.com/sashwattanay/ML-Iris-flowers-project)
- **Description:** Classify the Iris flower dataset using softmax regression implemented from scratch with batch gradient descent and early stopping.
- **Key Highlights:**
  - Scaled features and one-hot encoded target labels.
  - Implemented softmax regression, cross-entropy loss, and early stopping.
  - Achieved a test accuracy of **76.67%** with effective prevention of overfitting.
  - Visualized training/validation loss curves and decision boundaries.
- **Skills Demonstrated:** Algorithm development, optimization, feature scaling, visualization.

### [4. Predicting California Housing Prices Using Support Vector Regression](https://github.com/sashwattanay/ML-SVM-regression)
- **Description:** Predict median house values in California districts using Support Vector Regression (SVR).
- **Key Highlights:**
  - Scaled features using `StandardScaler` for SVR performance.
  - Performed hyperparameter tuning with `RandomizedSearchCV` across `C`, `epsilon`, and `kernel`.
  - Achieved a Root Mean Squared Error (RMSE) of **0.5672** and a Mean Absolute Percentage Error (MAPE) of **20.40%**.
- **Skills Demonstrated:** Support vector machines, hyperparameter tuning, cross-validation, regression analysis.

### [5. Handwritten Digit Classification Using Ensemble Learning](https://github.com/sashwattanay/ML-Ensemble-Learning-Random-Forest)
- **Description:** Classify handwritten digits (0-9) from the MNIST dataset using an ensemble of Random Forest, Extra-Trees, and Support Vector Machine (SVM) classifiers.
- **Key Highlights:**
  - Tuned hyperparameters for Random Forest and Extra-Trees using `RandomizedSearchCV`.
  - Combined classifiers using both **hard voting** and **soft voting** ensembles.
  - Achieved a test accuracy of **98.05%** using the soft voting ensemble.
- **Skills Demonstrated:** Ensemble learning, hyperparameter tuning, soft voting, model evaluation.

### [6. MNIST Digit Classification with Neural Networks](https://github.com/sashwattanay/ML-Neural-Nets-MNIST)
- **Description:** Classify handwritten digits from the MNIST dataset using a fully connected neural network.
- **Key Highlights:**
  - Hyperparameter tuning using **Keras Tuner's Hyperband**.
  - Early stopping and checkpointing to prevent overfitting.
  - Achieved a test accuracy of **97.58%** with well-tuned parameters.
  - Visualized training and validation performance through loss/accuracy plots.
- **Skills Demonstrated:** Neural networks, hyperparameter tuning, model evaluation, and diagnostics.


### [7. CIFAR-10 Image Classification with Deep Neural Networks](https://github.com/sashwattanay/ML-Deep-Learning-CIFAR10)
- **Description:** Classify images from the CIFAR-10 dataset using a deep multi-layer perceptron (MLP) with 20 hidden layers. The model incorporates Batch Normalization, L2 Regularization, Learning Rate Scheduling, and Early Stopping to combat overfitting and optimize performance.
- **Key Highlights:**
  - **Data Splitting:** The CIFAR-10 dataset was divided into training (80%), validation (20%), and test sets.
  - **Network Architecture:** A fully connected network with 20 hidden layers (100 neurons each) using the Swish activation function and He-normal initialization.
  - **Regularization & Optimization:** Integrated Batch Normalization and L2 regularization, with a custom learning rate scheduler and early stopping to restore the best model weights.
  - **Performance:** Achieved a test accuracy of approximately **50%**.
- **Skills Demonstrated:** Deep neural network design, data preprocessing, hyperparameter tuning, regularization techniques, and model evaluation.

## About Me
I am Sashwat Tanay, a theoretical astrophysicist by training, segueing into machine learning.
- **Personal Website:** [here](https://sashwattanay.github.io/site)
- **Kaggle Profile:** [here](https://www.kaggle.com/sashwattanay)

## How to Use
Click on the links above to explore individual projects. Each repository contains:
- A detailed `README.md` describing the project.
- Jupyter notebooks with code and documentation.
- Supporting files for data and dependencies.

## Contact
Feel free to connect with me:
- **Email:** [sashwattanay@gmail.com]
- **LinkedIn:** [https://www.linkedin.com/in/sashwat-tanay-22b13b214/]
