from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Reuse the dummy data generated for classification (X_cls, y_cls)
# X_train_cls, X_test_cls, y_train_cls, y_test_cls are already defined from the Logistic Regression example

# Create a K-Nearest Neighbors model
# Let's start with n_neighbors = 5 (a common default)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train_cls, y_train_cls)

# Make predictions
y_pred_knn = knn_model.predict(X_test_cls)

# Evaluate the model
accuracy_knn = accuracy_score(y_test_cls, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.2f}")
print("\nKNN Classification Report:")
print(classification_report(y_test_cls, y_pred_knn))

# Plot the decision boundary (for 2 features)
plt.figure(figsize=(8, 6))
plt.scatter(X_cls[:, 0], X_cls[:, 1], c=y_cls, cmap='viridis', edgecolors='k', marker='o', s=50)

# Get the min and max for the plot
x_min, x_max = X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1
y_min, y_max = X_cls[:, 1].min() - 1, X_cls[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z_knn = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_knn = Z_knn.reshape(xx.shape)

plt.contourf(xx, yy, Z_knn, alpha=0.3, cmap='viridis')
plt.title('K-Nearest Neighbors Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
