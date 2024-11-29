import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegressionEP34
from generate_dataset import generate_binary_problem

# Convert centers to a numpy array
centers = np.array([[0, 8], [0, 8]])

# Generate dataset
X, y = generate_binary_problem(centers, N=1000)

# Split the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the logistic regression model
model = LogisticRegressionEP34()

# Train the model
model.fit(X_train, y_train, iterations=10000, batch_size=None, show_step=1000, show_line=True)

# Predict on the test set
y_pred = model.predict(X_test) >= 0.5

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Observations:
# - During training, observe the printed loss values and the decision boundary if `show_line` is implemented.
# - Note how the loss decreases and the decision boundary changes as training progresses.
# - Try different centers for the classes to see how the model performs on more challenging problems.
