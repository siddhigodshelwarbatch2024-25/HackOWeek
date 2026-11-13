bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

bagging_model.fit(X_train, y_train)

bagging_train_pred = bagging_model.predict(X_train)
bagging_test_pred = bagging_model.predict(X_test)

print("Bagging Training Accuracy:",
      accuracy_score(y_train, bagging_train_pred))
print("Bagging Testing Accuracy:",
      accuracy_score(y_test, bagging_test_pred))
