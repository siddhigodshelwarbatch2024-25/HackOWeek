depths = range(1, 16)
train_scores = []
test_scores = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_scores.append(
        accuracy_score(y_train, model.predict(X_train))
    )
    test_scores.append(
        accuracy_score(y_test, model.predict(X_test))
    )

plt.figure(figsize=(9, 5))
plt.plot(depths, train_scores, marker="o", label="Training")
plt.plot(depths, test_scores, marker="s", label="Testing")
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Underfitting and Overfitting")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
