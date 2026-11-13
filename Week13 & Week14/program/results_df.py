models = {
    "Bagging": bagging_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

results = []

for name, model in models.items():
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1 Score": f1_score(y_test, test_pred)
    })

results_df = pd.DataFrame(results).round(4)
display(results_df)
