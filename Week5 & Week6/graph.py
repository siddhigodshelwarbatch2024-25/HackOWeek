import matplotlib.pyplot as plt

losses = []

for epoch in range(epochs):

    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)

    loss = -np.mean(
        y * np.log(predictions + 1e-8)
        + (1 - y) * np.log(1 - predictions + 1e-8)
    )

    losses.append(loss)

    dz = predictions - y
    dw = np.dot(X.T, dz) / len(X)
    db = np.mean(dz)

    weights -= learning_rate * dw
    bias -= learning_rate * db

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
