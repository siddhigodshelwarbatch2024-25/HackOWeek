X, y = fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True,
    as_frame=False
)

y = y.astype(int)

print("Dataset shape:", X.shape)
