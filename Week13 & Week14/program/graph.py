results_df.set_index("Model")[
    ["Train Accuracy", "Test Accuracy"]
].plot(kind="bar", figsize=(9, 5))

plt.title("Training vs Testing Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.01)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
