regularization_settings = {
    "No regularization": (0, 0),
    "L1 regularization": (5, 0),
    "L2 regularization": (0, 10),
    "L1 + L2": (5, 10)
}

reg_results = []

for name, (alpha, lam) in regularization_settings.items():
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        reg_alpha=alpha,
        reg_lambda=lam,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, model.predict(X_train)
    )
    test_acc = accuracy_score(
        y_test, model.predict(X_test)
    )

    reg_results.append({
        "Regularization": name,
        "Training Accuracy": train_acc,
        "Testing Accuracy": test_acc,
        "Generalization Gap": train_acc - test_acc
    })

display(pd.DataFrame(reg_results).round(4))
