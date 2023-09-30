import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a dataset (Iris dataset as an example)
data = load_iris()
X = data.data
y = data.target

# Split the data into a small initial labeled set and a larger pool of unlabeled data
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.9, random_state=42)

# Create a random forest classifier
classifier = RandomForestClassifier(n_estimators=100)

# Active learning loop
for i in range(5):  # You can specify the number of iterations
    # Train the classifier on the labeled data
    classifier.fit(X_labeled, y_labeled)

    # Predict on the unlabeled data
    y_pred_unlabeled = classifier.predict(X_unlabeled)

    # Calculate uncertainty (e.g., using entropy) for each prediction
    uncertainty = -np.max(classifier.predict_proba(X_unlabeled), axis=1)

    # Select the most uncertain data points
    num_points_to_label = 10  # You can adjust this number
    uncertain_indices = np.argsort(uncertainty)[-num_points_to_label:]

    # Add the selected data points to the labeled set
    X_labeled = np.vstack((X_labeled, X_unlabeled[uncertain_indices]))
    y_labeled = np.hstack((y_labeled, y_pred_unlabeled[uncertain_indices]))

    # Remove the newly labeled data points from the unlabeled pool
    X_unlabeled = np.delete(X_unlabeled, uncertain_indices, axis=0)

# Evaluate the final model
y_test = data.target
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final model accuracy: {accuracy:.2f}")
