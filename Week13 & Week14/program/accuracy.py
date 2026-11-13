experiments = {
    "Underfitting (depth=1)": DecisionTreeClassifier(
        max_depth=1, random_state=42
    ),
    "Moderate complexity (depth=4)": DecisionTreeClassifier(
        max_depth=4, random_state=42
    ),
    "High complexity (unrestricted)": DecisionTreeClassifier(
        random_state=42
    )
}

for name, model in experiments.items():
    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, model.predict(X_train)
    )
    test_acc = accuracy_score(
        y_test, model.predict(X_test)
    )

    print(f"\n{name}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy:  {test_acc:.4f}")
    print(f"Accuracy gap:      {train_acc - test_acc:.4f}")
