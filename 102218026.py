import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score

# Load the dataset (digits dataset in this example, replace with your dataset as needed)
dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

# Split the dataset into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for NuSVC
param_grid = {
    'nu': [0.1, 0.5, 0.9],   # Nu parameter for NuSVC
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],     # Degree of the polynomial kernel function
}

# Create a NuSVC classifier
nusvc = NuSVC()

# Use GridSearchCV to perform parameter optimization
grid_search = GridSearchCV(estimator=nusvc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters and accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Test the model on the test data
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Output the results
print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_accuracy)
print("Test Set Accuracy:", test_accuracy)

# Plotting convergence graph for best NuSVC over iterations
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']
iterations = range(1, len(mean_test_scores) + 1)

plt.plot(iterations, mean_test_scores, marker='o')
plt.title("Convergence Graph of NuSVC Optimization")
plt.xlabel("Iterations")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True)
plt.show()