best_model_name = results_df.loc[
    results_df["Test Accuracy"].idxmax(), "Model"
]
best_model = models[best_model_name]

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_model.predict(X_test),
    display_labels=data.target_names
)

plt.title(f"Confusion Matrix: {best_model_name}")
plt.tight_layout()
plt.show()

print("Selected model:", best_model_name)
print(classification_report(
    y_test,
    best_model.predict(X_test),
    target_names=data.target_names
))
