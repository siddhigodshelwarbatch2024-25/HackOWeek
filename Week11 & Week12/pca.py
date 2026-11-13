pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("PCA shape:", X_pca.shape)
print("Explained variance:", pca.explained_variance_ratio_)
print("Total variance:",
      pca.explained_variance_ratio_.sum())
