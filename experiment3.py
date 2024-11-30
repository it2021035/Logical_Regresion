import logistic_regression as lr
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Parameters
n_repeats = 20

# Store accuracies
accuracies = []

# Start timing
start_time = time.time()

for _ in range(n_repeats):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
    
    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# End timing
end_time = time.time()

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Get system information
import platform
import psutil

processor = platform.processor()
memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
execution_time = end_time - start_time

# Print results
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Processor: {processor}")
print(f"Memory: {memory:.2f} GB")

