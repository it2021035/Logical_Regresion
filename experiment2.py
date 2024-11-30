import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from logistic_regression import LogisticRegressionEP34
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Parameters
n_repeats = 20
learning_rate = 0.01

# Store accuracies
accuracies = [] 

# initialize model
model = LogisticRegressionEP34()

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
    model.fit(X_train, y_train, iterations=10000, batch_size=64, show_step=1000, show_line=False)
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)  # Convert probabilities to binary class labels
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
