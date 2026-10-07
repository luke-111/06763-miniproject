import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# 1. Load datasets
df_normal = pd.read_parquet('data/tep_fault_free_training.parquet')
df_faulty = pd.read_parquet('data/tep_faulty_training_runs01-20.parquet')

# 2. Define 52 observation channels (xmeas_1 to 41, xmv_1 to 11)
channels = [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)]

# Assign faultNumber 0 to fault-free data as per project requirements
df_normal['faultNumber'] = 0

# 3. Split data into training and scoring sets
# Training: Fault-free runs 1 to 300
train_mask = (df_normal['simulationRun'] >= 1) & (df_normal['simulationRun'] <= 300)
df_train = df_normal[train_mask]

# Scoring: Validation (301-400), Test (401-500), and all faulty runs (1-20)
score_mask = (df_normal['simulationRun'] >= 301) & (df_normal['simulationRun'] <= 500)
df_score = pd.concat([df_normal[score_mask], df_faulty], ignore_index=True)

# 4. Standardize data
# Compute mean and std (ddof=1) exclusively on the training set to prevent data leakage
train_mean = df_train[channels].mean()
train_std = df_train[channels].std(ddof=1)

Z_train = (df_train[channels] - train_mean) / train_std
Z_score = (df_score[channels] - train_mean) / train_std

# 5. Determine the number of principal components (k) for 90% variance threshold
pca_initial = PCA()
pca_initial.fit(Z_train)

cum_var = np.cumsum(pca_initial.explained_variance_ratio_)
k = np.argmax(cum_var >= 0.90) + 1
print(f"Number of principal components (k) retaining >= 90% variance: {k}")

# Output the exact cumulative variance for k-1 and k to validate the threshold
if k > 1:
    print(f" - {k - 1} components: {cum_var[k - 2] * 100:.4f}% cumulative explained variance")
print(f" - {k} components: {cum_var[k - 1] * 100:.4f}% cumulative explained variance")

# 6. Fit the final PCA model and compute monitoring statistics
pca = PCA(n_components=k)
pca.fit(Z_train)

# Compute scores (t = zP) and eigenvalues (lambda)
t_scores = pca.transform(Z_score)
lam = pca.explained_variance_

# Calculate Hotelling's T^2 statistic
T2 = np.sum((t_scores ** 2) / lam, axis=1)

# Calculate Squared Prediction Error (SPE)
z_reconstructed = pca.inverse_transform(t_scores)
SPE = np.sum((Z_score.values - z_reconstructed) ** 2, axis=1)

# 7. Format and save the final results
df_score['T2'] = T2
df_score['SPE'] = SPE

final_cols = ['faultNumber', 'simulationRun', 'sample', 'T2', 'SPE']
df_final = df_score[final_cols]

os.makedirs('results', exist_ok=True)
df_final.to_parquet('results/scores_pca.parquet', index=False)
print("PCA monitoring scores successfully saved to 'results/scores_pca.parquet'")