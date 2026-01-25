# %%
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import h5py
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.file_adapters import ArrayRecordFileAdapter
from tensorflow_datasets.core.example_parser import ExampleParser
from tensorflow_datasets.core import features as tfds_features
import json

# %%
import json
from array_record.python.array_record_data_source import ArrayRecordDataSource

path = "/workspace/data/clic_edm_qq_pf-test.array_record-00000-of-00008"

# Load features.json if you have it
with open("features.json", "r") as f:
    features_info = json.load(f)
    print("Features:", features_info)

# Read the array records
with ArrayRecordDataSource(path) as ds:
    print(f"Dataset has {len(ds)} records")

    # Get first record (it's bytes)
    raw_bytes = ds[0]

    # If it's a serialized format, you'll need to deserialize it
    # For example, if it's TensorFlow Example format:
    # import tensorflow as tf
    # example = tf.train.Example()
    # example.ParseFromString(raw_bytes)

# %%

path = "/workspace/clic_edm_qq_pf-test.array_record-00000-of-00008"

# Load features info to understand the structure
with open("features.json", "r") as f:
    features_info = json.load(f)

# Read and decode records
with ArrayRecordDataSource(path) as ds:
    # Get first record
    raw_bytes = ds[0]

    # Parse as TensorFlow Example
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)

    # Convert to dict
    features = example.features.feature

    print("Available features:")
    for key in features.keys():
        print(f"  - {key}")

    # Access specific features
    # For float tensors:
    if 'X' in features:
        x_data = features['X'].float_list.value
        print(f"\nX shape: {len(x_data)}")
        print(f"X first 10 values: {list(x_data[:10])}")

    if 'ycand' in features:
        ycand_data = features['ycand'].float_list.value
        print(f"\nycand shape: {len(ycand_data)}")
        print(f"ycand first 10 values: {list(ycand_data[:10])}")

    # For other features, check their type and decode accordingly
    for key, value in features.items():
        if value.HasField('float_list'):
            print(f"{key}: float array of length {len(value.float_list.value)}")
        elif value.HasField('int64_list'):
            print(f"{key}: int64 array of length {len(value.int64_list.value)}")
        elif value.HasField('bytes_list'):
            print(f"{key}: bytes array of length {len(value.bytes_list.value)}")

# %%

path = "/workspace/data/clic_edm_qq_pf-test.array_record-00000-of-00008"

with ArrayRecordDataSource(path) as ds:
    raw_bytes = ds[0]
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)

    print("Feature sizes:")
    total = 0
    for key, feature in example.features.feature.items():
        size = len(feature.float_list.value)
        total += size
        print(f"  {key:15s}: {size:5d} values")

    print(f"\nTotal values per record: {total}")
    print(f"Total records in dataset: {len(ds)}")
    print(f"Total values in entire dataset: {total * len(ds):,}")

# %%

path = "/workspace/clic_edm_qq_pf-test.array_record-00000-of-00008"

with ArrayRecordDataSource(path) as ds:
    raw_bytes = ds[0]
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)

    X = np.array(example.features.feature['X'].float_list.value)

    print(f"X shape: {X.shape}")
    print(f"X min: {X.min()}, max: {X.max()}")

    # Check if there's a pattern - particle physics data often has structure
    # Common pattern: [num_particles, features_per_particle]

    # Let's check possible factorizations of 1309
    print("\nPossible interpretations:")
    for n_particles in [127, 131, 1309]:
        if 1309 % n_particles == 0:
            n_features = 1309 // n_particles
            print(f"  {n_particles} particles × {n_features} features each")

    # Check dataset_info.json if available
    print("\nLet's check dataset_info.json for the actual structure:")

# %%

# Find all test shards
data_dir = Path("/workspace/data")
test_shards = sorted(data_dir.glob("clic_edm_qq_pf-test.array_record-*"))
print(f"Found {len(test_shards)} shards:")
for shard in test_shards:
    print(f"  {shard.name}")


def decode_example(raw_bytes):
    """Decode a single example and extract X (input particles)."""
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)

    # Get X features (input particles)
    X = np.array(
        example.features.feature['X'].float_list.value, dtype=np.float32)

    # Reshape to [num_particles, 17]
    if len(X) > 0:
        X = X.reshape(-1, 17)
    else:
        X = np.array([]).reshape(0, 17)

    return X


# First pass: count total particles across all shards
print("\nCounting total particles...")
total_particles = 0
total_events = 0

for shard_path in tqdm(test_shards, desc="Counting"):
    with ArrayRecordDataSource(str(shard_path)) as ds:
        total_events += len(ds)
        for i in range(len(ds)):
            X = decode_example(ds[i])
            total_particles += len(X)

print(f"\nTotal events: {total_events:,}")
print(f"Total particles: {total_particles:,}")
print(f"Average particles per event: {total_particles / total_events:.1f}")

# Second pass: load all particles into one array
print("\nLoading all particles...")
all_particles = np.zeros((total_particles, 17), dtype=np.float32)
particle_idx = 0

for shard_path in tqdm(test_shards, desc="Loading"):
    with ArrayRecordDataSource(str(shard_path)) as ds:
        for i in range(len(ds)):
            X = decode_example(ds[i])
            num_particles = len(X)
            all_particles[particle_idx:particle_idx + num_particles] = X
            particle_idx += num_particles

print(f"\nLoaded {particle_idx:,} particles")

# Save to HDF5 (efficient binary format)
output_path = "/workspace/data/clic_test_particles.h5"
print(f"\nSaving to {output_path}...")

with h5py.File(output_path, 'w') as f:
    f.create_dataset('particles', data=all_particles,
                     compression='gzip', compression_opts=9)

    # Store metadata
    f.attrs['num_particles'] = total_particles
    f.attrs['num_events'] = total_events
    f.attrs['num_features'] = 17
    f.attrs['avg_particles_per_event'] = total_particles / total_events
    f.attrs['feature_names'] = [
        'elemtype', 'pt_or_et', 'eta', 'sin_phi', 'cos_phi',
        'p_or_energy', 'chi2_or_pos_x', 'ndf_or_pos_y', 'dEdx_or_pos_z',
        'dEdxError_or_iTheta', 'radiusOfInnermostHit_or_energy_ecal',
        'tanLambda_or_energy_hcal', 'D0_or_energy_other', 'omega_or_num_hits',
        'Z0_or_sigma_x', 'time_or_sigma_y', 'sigma_z'
    ]

print(f"\nSaved! File size: {os.path.getsize(output_path) / 1e9:.2f} GB")

# Print statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Output file: {output_path}")
print(f"Total particles (data points): {total_particles:,}")
print(f"Features per particle: 17")
print(f"Shape: ({total_particles:,}, 17)")
print(f"Data type: float32")
print(f"Memory size: {all_particles.nbytes / 1e9:.2f} GB")
print(f"Disk size (compressed): {os.path.getsize(output_path) / 1e9:.2f} GB")
print(f"\nOriginal events: {total_events:,}")
print(f"Avg particles per event: {total_particles / total_events:.1f}")
print("\nFeature statistics:")
for i, name in enumerate(['elemtype', 'pt_or_et', 'eta', 'sin_phi', 'cos_phi']):
    col = all_particles[:, i]
    print(
        f"  {name:20s}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")

print("\n" + "="*60)

# Test loading it back
print("\nTesting reload...")
with h5py.File(output_path, 'r') as f:
    loaded_particles = f['particles'][:]
    print(f"Loaded shape: {loaded_particles.shape}")
    print(f"First particle: {loaded_particles[0]}")
    print(f"\nMetadata:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")

print("\n✓ Done! Use this to load later:")
print(f"```python")
print(f"import h5py")
print(f"with h5py.File('{output_path}', 'r') as f:")
print(f"    particles = f['particles'][:]  # shape: ({total_particles}, 17)")
print(f"```")

# %%
"""
Feature Analysis for k-NN Dimensionality Selection
Run this after creating the HDF5 file
"""
matplotlib.use('Agg')  # Non-interactive backend

# Load data
print("Loading data...")
HDF5_PATH = "/workspace/data/clic_test_particles.h5"

with h5py.File(HDF5_PATH, 'r') as f:
    X = f['particles'][:]
    feature_names = [s.decode() if isinstance(
        s, bytes) else s for s in f.attrs['feature_names']]

print(f"Loaded {X.shape[0]:,} particles with {X.shape[1]} features\n")

# Sample for faster analysis
np.random.seed(42)
sample_size = min(100000, len(X))
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_idx]

print(f"Using sample of {sample_size:,} particles for analysis\n")

# ============================================================
# 1. BASIC STATISTICS
# ============================================================
print("="*80)
print("1. BASIC FEATURE STATISTICS")
print("="*80)

stats_list = []
for i, name in enumerate(feature_names):
    col = X_sample[:, i]
    stats_list.append({
        'Feature': name,
        'Min': col.min(),
        'Max': col.max(),
        'Mean': col.mean(),
        'Std': col.std(),
        'Range': col.max() - col.min()
    })

stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False))

# ============================================================
# 2. VARIANCE ANALYSIS
# ============================================================
print("\n" + "="*80)
print("2. VARIANCE ANALYSIS (after standardization)")
print("="*80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

variance = X_scaled.var(axis=0)
variance_list = [{'Feature': name, 'Variance': var}
                 for name, var in zip(feature_names, variance)]
variance_df = pd.DataFrame(variance_list).sort_values(
    'Variance', ascending=False)

print(variance_df.to_string(index=False))
print(f"\nTop 5 features by variance:")
for i, row in variance_df.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Variance']:.4f}")

# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*80)
print("3. CORRELATION ANALYSIS")
print("="*80)

corr_matrix = np.corrcoef(X_sample.T)

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(corr_matrix[i, j]) > 0.7:
            high_corr_pairs.append(
                (feature_names[i], feature_names[j], corr_matrix[i, j]))

if high_corr_pairs:
    print("\nHighly correlated feature pairs (|r| > 0.7):")
    for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {f1:30s} <-> {f2:30s}: {corr:6.3f}")
else:
    print("\nNo highly correlated feature pairs found (|r| > 0.7)")

# ============================================================
# 4. PCA ANALYSIS
# ============================================================
print("\n" + "="*80)
print("4. PCA ANALYSIS")
print("="*80)

pca = PCA()
pca.fit(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("\nExplained variance by principal component:")
for i in range(min(10, len(explained_var))):
    print(
        f"  PC{i+1:2d}: {explained_var[i]*100:5.2f}% (cumulative: {cumulative_var[i]*100:6.2f}%)")

print(f"\nVariance explained by top 3 PCs: {cumulative_var[2]*100:.2f}%")
print(f"Variance explained by top 5 PCs: {cumulative_var[4]*100:.2f}%")

print("\nTop contributing features to first 3 PCs:")
for pc_idx in range(3):
    loadings = pca.components_[pc_idx]
    top_idx = np.argsort(np.abs(loadings))[-5:][::-1]
    print(
        f"\n  PC{pc_idx+1} (explains {explained_var[pc_idx]*100:.2f}% variance):")
    for idx in top_idx:
        print(f"    {feature_names[idx]:30s}: {loadings[idx]:7.3f}")

# ============================================================
# 5. TRACK vs CLUSTER ANALYSIS
# ============================================================
print("\n" + "="*80)
print("5. TRACK vs CLUSTER DISCRIMINATION")
print("="*80)

is_cluster = (X_sample[:, 0] == 2)
n_tracks = (~is_cluster).sum()
n_clusters = is_cluster.sum()

print(f"\nDataset composition:")
print(f"  Tracks:   {n_tracks:,} ({n_tracks/len(X_sample)*100:.1f}%)")
print(f"  Clusters: {n_clusters:,} ({n_clusters/len(X_sample)*100:.1f}%)")

print(f"\n{'Feature':<30s} {'Track Mean':>12s} {'Cluster Mean':>14s} {'|Diff|':>10s}")
print("-" * 70)
discriminative_features = []
for i, name in enumerate(feature_names):
    if i == 0:  # Skip elemtype itself
        continue
    track_mean = X_sample[~is_cluster, i].mean()
    cluster_mean = X_sample[is_cluster, i].mean()
    diff = abs(track_mean - cluster_mean)
    discriminative_features.append((name, diff))
    print(f"{name:<30s} {track_mean:12.3f} {cluster_mean:14.3f} {diff:10.3f}")

# Sort by discrimination power
discriminative_features.sort(key=lambda x: x[1], reverse=True)
print(f"\nMost discriminative features (largest difference):")
for i, (name, diff) in enumerate(discriminative_features[:5]):
    print(f"  {i+1}. {name}: {diff:.3f}")

# ============================================================
# 6. RECOMMENDATIONS
# ============================================================
print("\n" + "="*80)
print("6. FEATURE SELECTION RECOMMENDATIONS FOR k-NN")
print("="*80)

recommendations = {}

# Option 1: Kinematic 5D
print("\n" + "-"*80)
print("OPTION 1: KINEMATIC FEATURES (5D) - RECOMMENDED")
print("-"*80)
kinematic_5d = ['pt_or_et', 'eta', 'sin_phi', 'cos_phi', 'p_or_energy']
print(f"Features: {', '.join(kinematic_5d)}")
print("\nWhy:")
print("  • Physics-motivated: These are the fundamental kinematic variables")
print("  • pt/energy: How energetic the particle is")
print("  • eta: Angular position (forward/backward)")
print("  • sin_phi, cos_phi: Azimuthal direction (avoids 2π wraparound)")
print("  • p_or_energy: Total momentum/energy")
print("\nUse case: Best for general particle similarity")
print("Pros: Interpretable, physically meaningful, captures particle identity")
print("Cons: Doesn't include detector-specific info")

recommendations['kinematic_5d'] = {
    'features': kinematic_5d,
    'indices': [1, 2, 3, 4, 5],
    'description': 'Physics-motivated kinematic variables'
}

# Option 2: Kinematic 3D
print("\n" + "-"*80)
print("OPTION 2: MINIMAL KINEMATIC (3D)")
print("-"*80)
kinematic_3d = ['pt_or_et', 'eta', 'p_or_energy']
print(f"Features: {', '.join(kinematic_3d)}")
print("\nWhy:")
print("  • Core kinematic variables only")
print("  • Removes angular coordinates (keeps just magnitude and eta)")
print("\nUse case: Fast k-NN, when you don't care about azimuthal direction")
print("Pros: Fastest computation, still physically meaningful")
print("Cons: Loses directional information")

recommendations['kinematic_3d'] = {
    'features': kinematic_3d,
    'indices': [1, 2, 5],
    'description': 'Minimal kinematic set'
}

# Option 3: High variance 5D
top_5_var = variance_df.head(5)['Feature'].tolist()
print("\n" + "-"*80)
print("OPTION 3: HIGH VARIANCE FEATURES (5D)")
print("-"*80)
print(f"Features: {', '.join(top_5_var)}")
print("\nWhy:")
print("  • Data-driven selection based on variance")
print("  • Maximizes information content")
print(f"  • Combined variance: {variance_df.head(5)['Variance'].sum():.2f}")
print("\nUse case: When you want maximum data spread")
print("Pros: Captures most variation in data")
print("Cons: May include correlated features, less interpretable")

recommendations['high_variance_5d'] = {
    'features': top_5_var,
    'indices': [feature_names.index(f) for f in top_5_var],
    'description': 'Top 5 features by variance'
}

# Option 4: PCA
print("\n" + "-"*80)
print("OPTION 4: PCA PROJECTION (3D or 5D)")
print("-"*80)
print("Use principal components instead of original features")
print(f"\n3D PCA: Captures {cumulative_var[2]*100:.2f}% of variance")
print(f"5D PCA: Captures {cumulative_var[4]*100:.2f}% of variance")
print("\nWhy:")
print("  • Optimal linear combination for variance")
print("  • Removes correlation between features")
print("  • Maximum information in minimum dimensions")
print("\nUse case: When you want mathematical optimality")
print("Pros: Optimal dimensionality reduction, uncorrelated features")
print("Cons: Less interpretable, requires transformation of new data")

recommendations['pca_3d'] = {
    'features': ['PC1', 'PC2', 'PC3'],
    'description': f'Top 3 PCs ({cumulative_var[2]*100:.1f}% variance)',
    'transform': 'pca',
    'variance_explained': float(cumulative_var[2])
}

recommendations['pca_5d'] = {
    'features': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
    'description': f'Top 5 PCs ({cumulative_var[4]*100:.1f}% variance)',
    'transform': 'pca',
    'variance_explained': float(cumulative_var[4])
}

# ============================================================
# 7. FINAL RECOMMENDATION
# ============================================================
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

print("\n🎯 For particle physics k-NN, I recommend:")
print("\n  ★ 5D KINEMATIC FEATURES ★")
print("    [pt_or_et, eta, sin_phi, cos_phi, p_or_energy]")
print("\n  Reasons:")
print("    1. Physically meaningful and interpretable")
print("    2. Captures particle identity (momentum + direction)")
print("    3. No highly correlated features")
print("    4. Standard in particle physics")
print("\n  If you need 3D, use:")
print("    [pt_or_et, eta, p_or_energy]")

# Check correlation within recommended features
print("\n  Correlation matrix for recommended 5D features:")
kinematic_idx = [1, 2, 3, 4, 5]
corr_kinematic = np.corrcoef(X_sample[:, kinematic_idx].T)
print(f"  {'':12s}", end='')
for feat in kinematic_5d:
    print(f"{feat:12s}", end='')
print()
for i, feat_i in enumerate(kinematic_5d):
    print(f"  {feat_i:12s}", end='')
    for j in range(len(kinematic_5d)):
        print(f"{corr_kinematic[i, j]:12.3f}", end='')
    print()

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save recommendations
with open('/mnt/user-data/outputs/feature_recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)
print("✓ Saved: /mnt/user-data/outputs/feature_recommendations.json")

# Save PCA transformer
with open('/mnt/user-data/outputs/pca_transformer.pkl', 'wb') as f:
    pickle.dump({'pca': pca, 'scaler': scaler}, f)
print("✓ Saved: /mnt/user-data/outputs/pca_transformer.pkl")

# Create visualizations
print("\nGenerating visualizations...")
fig = plt.figure(figsize=(20, 15))

# 1. Variance bar plot
ax1 = plt.subplot(3, 3, 1)
var_plot = variance_df.head(10)
ax1.barh(range(len(var_plot)), var_plot['Variance'], color='steelblue')
ax1.set_yticks(range(len(var_plot)))
ax1.set_yticklabels(var_plot['Feature'], fontsize=9)
ax1.set_xlabel('Variance (standardized)', fontsize=10)
ax1.set_title('Top 10 Features by Variance', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 2. PCA scree plot
ax2 = plt.subplot(3, 3, 2)
ax2.bar(range(1, 11), explained_var[:10],
        color='steelblue', alpha=0.7, label='Individual')
ax2.plot(range(1, 11), cumulative_var[:10], 'ro-',
         linewidth=2, markersize=6, label='Cumulative')
ax2.set_xlabel('Principal Component', fontsize=10)
ax2.set_ylabel('Explained Variance Ratio', fontsize=10)
ax2.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80%')
ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')

# 3. Correlation heatmap (top features)
ax3 = plt.subplot(3, 3, 3)
top_features = variance_df.head(8)['Feature'].tolist()
top_idx = [feature_names.index(f) for f in top_features]
corr_subset = corr_matrix[np.ix_(top_idx, top_idx)]
im = ax3.imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax3.set_xticks(range(len(top_features)))
ax3.set_yticks(range(len(top_features)))
ax3.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
ax3.set_yticklabels(top_features, fontsize=8)
ax3.set_title('Correlation Matrix (Top 8 Features)',
              fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax3)

# 4. pt_or_et distribution
ax4 = plt.subplot(3, 3, 4)
pt_data = X_sample[:, 1]
pt_data = pt_data[pt_data > 0]
ax4.hist(np.log10(pt_data), bins=50, color='steelblue',
         alpha=0.7, edgecolor='black')
ax4.set_xlabel('log10(pt_or_et)', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('Transverse Momentum Distribution',
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. eta distribution
ax5 = plt.subplot(3, 3, 5)
eta_data = X_sample[:, 2]
ax5.hist(eta_data, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax5.set_xlabel('eta (pseudorapidity)', fontsize=10)
ax5.set_ylabel('Count', fontsize=10)
ax5.set_title('Eta Distribution', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. 2D scatter: pt vs eta colored by type
ax6 = plt.subplot(3, 3, 6)
sample_plot = np.random.choice(len(X_sample), 5000, replace=False)
pt_plot = X_sample[sample_plot, 1]
eta_plot = X_sample[sample_plot, 2]
type_plot = X_sample[sample_plot, 0]
scatter = ax6.scatter(eta_plot, np.log10(pt_plot + 1),
                      c=type_plot, alpha=0.5, s=5, cmap='viridis')
ax6.set_xlabel('eta', fontsize=10)
ax6.set_ylabel('log10(pt_or_et)', fontsize=10)
ax6.set_title('Particle Distribution: pt vs eta',
              fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('elemtype', fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. PCA 2D projection
ax7 = plt.subplot(3, 3, 7)
X_pca = pca.transform(X_scaled)
sample_plot = np.random.choice(len(X_pca), 5000, replace=False)
scatter = ax7.scatter(X_pca[sample_plot, 0], X_pca[sample_plot, 1],
                      c=X_sample[sample_plot, 0], alpha=0.5, s=5, cmap='viridis')
ax7.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=10)
ax7.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=10)
ax7.set_title('PCA Projection (2D)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax7, label='elemtype')
ax7.grid(True, alpha=0.3)

# 8. sin_phi vs cos_phi (should be circular)
ax8 = plt.subplot(3, 3, 8)
sample_plot = np.random.choice(len(X_sample), 5000, replace=False)
ax8.scatter(X_sample[sample_plot, 3], X_sample[sample_plot, 4],
            alpha=0.3, s=3, c=X_sample[sample_plot, 0], cmap='viridis')
ax8.set_xlabel('sin_phi', fontsize=10)
ax8.set_ylabel('cos_phi', fontsize=10)
ax8.set_title('Phi Distribution (should be circular)',
              fontsize=12, fontweight='bold')
ax8.set_xlim(-1.1, 1.1)
ax8.set_ylim(-1.1, 1.1)
ax8.set_aspect('equal')
circle = plt.Circle((0, 0), 1, fill=False, color='red',
                    linestyle='--', linewidth=2)
ax8.add_patch(circle)
ax8.grid(True, alpha=0.3)

# 9. Feature importance summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
RECOMMENDED FOR k-NN:

5D (BEST):
  • pt_or_et
  • eta
  • sin_phi
  • cos_phi  
  • p_or_energy

3D (FAST):
  • pt_or_et
  • eta
  • p_or_energy

Dataset Size:
  • {X.shape[0]:,} particles
  • 17 features each
  
Why these features?
  ✓ Physically meaningful
  ✓ Low correlation
  ✓ Captures particle ID
  ✓ Standard in HEP
"""
ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/feature_analysis.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: /mnt/user-data/outputs/feature_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review the visualization: feature_analysis.png")
print("2. Choose your feature set (recommended: kinematic_5d)")
print("3. Extract those features for k-NN")
print("\nExample code:")
print("```python")
print("import h5py")
print(f"with h5py.File('{HDF5_PATH}', 'r') as f:")
print("    X_full = f['particles'][:]")
print("# Use kinematic 5D features")
print("X_knn = X_full[:, [1, 2, 3, 4, 5]]  # pt, eta, sin_phi, cos_phi, p")
print("print(f'k-NN input shape: {X_knn.shape}')  # (5.6M, 5)")
print("```")

# %%
!pip install scikit-learn

# %%
