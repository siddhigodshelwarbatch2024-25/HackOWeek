lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=15,
    min_child_samples=20,
    reg_alpha=0.1,     # L1 regularization
    reg_lambda=1.0,    # L2 regularization
    verbosity=-1,
    random_state=42,
    n_jobs=-1
)

lgbm_model.fit(X_train, y_train)

lgbm_train_pred = lgbm_model.predict(X_train)
lgbm_test_pred = lgbm_model.predict(X_test)

print("LightGBM Training Accuracy:",
      accuracy_score(y_train, lgbm_train_pred))
print("LightGBM Testing Accuracy:",
      accuracy_score(y_test, lgbm_test_pred))
