xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,     # L1 regularization
    reg_lambda=1.0,    # L2 regularization
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

print("XGBoost Training Accuracy:",
      accuracy_score(y_train, xgb_train_pred))
print("XGBoost Testing Accuracy:",
      accuracy_score(y_test, xgb_test_pred))
