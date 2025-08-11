import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances , euclidean_distances
import seaborn as sns
from typing import List, Tuple
import os

class FeatureVisualizer:
    def __init__(self, output_dir: str = "result"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_diversity(self, 
                          features: List[np.ndarray], 
                          video_paths: List[str],
                          query: str,
                          selected_indices: List[int] = None) -> None:
        if len(features) == 0:
            print("No features to visualize")
            return
            
        feature_matrix = self._prepare_features(features)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Diversity Analysis - {query.upper()} Stroke', fontsize=16)
        
        self._plot_pca_2d(feature_matrix, video_paths, selected_indices, ax1, query)
        
        self._plot_distance_heatmap(feature_matrix, selected_indices, ax2)

        self._plot_feature_magnitudes(feature_matrix, selected_indices, ax3)
        
        self._plot_distance_distribution(feature_matrix, selected_indices, ax4)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'pca_{query}_features.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.show()
    def visualize_comparison(
        self,
        features: List[np.ndarray],
        query: str,
        diversity_indices: List[int],
        random_indices: List[int]
    ) -> None:
        """
        Draw a 2x2 grid:
          [0,0] PCA of diversity (green)
          [0,1] PCA of random    (red)
          [1,0] Pairwise Cosine Distances (diversity vs random)
          [1,1] Pairwise Euclidean Distances (diversity vs random)
        """
        # 1) Prepare feature matrix
        X = self._prepare_features(features)  # (N, D)

        # 2) PCA projection
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)

        # 3) Distance matrices
        cos_mat = cosine_distances(X)
        euc_mat = euclidean_distances(X)

        # 4) Setup subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Comparison: {query.upper()}", fontsize=16)

        # Top-left: PCA diversity
        ax = axes[0, 0]
        ax.scatter(X2[:,0], X2[:,1], c='lightgray', alpha=0.3, s=30)
        ax.scatter(X2[diversity_indices,0], X2[diversity_indices,1], c='green', s=80, label='Farthest‐First Sampling')
        ax.set_title('PCA: Farthest‐First Sampling')
        ax.legend(); ax.grid(alpha=0.3)

        # Top-right: PCA random
        ax = axes[0, 1]
        ax.scatter(X2[:,0], X2[:,1], c='lightgray', alpha=0.3, s=30)
        ax.scatter(X2[random_indices,0], X2[random_indices,1], c='red', s=80, label='Random')
        ax.set_title('PCA: Uniform Sampling')
        ax.legend(); ax.grid(alpha=0.3)

        # Bottom-left: Cosine distance distribution
        ax = axes[1, 0]
        all_pairs = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        div_pairs = [cos_mat[i,j] for i in diversity_indices for j in diversity_indices if i<j]
        rand_pairs = [cos_mat[i,j] for i in random_indices   for j in random_indices   if i<j]
        ax.hist(all_pairs,  bins=30, alpha=0.5, density=True, label='All pairs')
        ax.hist(div_pairs,  bins=15, alpha=0.7, color='green', density=True, label='Farthest‐First Sampling')
        ax.hist(rand_pairs, bins=15, alpha=0.7, color='red',   density=True, label='Uniform Sampling')
        ax.axvline(np.mean(all_pairs), color='blue', linestyle='--',
        label=f'Mean all: {np.mean(all_pairs):.3f}')
        ax.axvline(np.mean(div_pairs), color='green', linestyle='--',
        label=f'Mean selected: {np.mean(div_pairs):.3f}')
        ax.axvline(np.mean(rand_pairs), color='red', linestyle='--',
        label=f'Mean selected: {np.mean(rand_pairs):.3f}')
        ax.set_title('Cosine Distance Distribution')
        ax.set_xlabel('Cosine Distance'); ax.set_ylabel('Density')
        ax.legend(); ax.grid(alpha=0.3)

        # Bottom-right: Euclidean distance distribution
        ax = axes[1, 1]
        all_pairs_e = euc_mat[np.triu_indices_from(euc_mat, k=1)]
        div_pairs_e = [euc_mat[i,j] for i in diversity_indices for j in diversity_indices if i<j]
        rand_pairs_e = [euc_mat[i,j] for i in random_indices   for j in random_indices   if i<j]
        ax.hist(all_pairs_e,  bins=30, alpha=0.5, density=True, label='All pairs')
        ax.hist(div_pairs_e,  bins=15, alpha=0.7, color='green', density=True, label='Farthest‐First Sampling')
        ax.hist(rand_pairs_e, bins=15, alpha=0.7, color='red',   density=True, label='Uniform Sampling')

        ax.axvline(np.mean(all_pairs_e), color='blue', linestyle='--',
        label=f'Mean all: {np.mean(all_pairs_e):.3f}')
        ax.axvline(np.mean(div_pairs_e), color='green', linestyle='--',
        label=f'Mean selected: {np.mean(div_pairs_e):.3f}')
        ax.axvline(np.mean(rand_pairs_e), color='red', linestyle='--',
        label=f'Mean selected: {np.mean(rand_pairs_e):.3f}')
        ax.set_title('Euclidean Distance Distribution')
        ax.set_xlabel('Euclidean Distance'); ax.set_ylabel('Density')
        ax.legend(); ax.grid(alpha=0.3)

        # 5) Save & show
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = os.path.join(self.output_dir, f'comparison_{query}.png')
        plt.savefig(out, dpi=300)
        print(f"Saved comparison figure to: {out}")
        plt.show()
    def _prepare_features(self, features: List[np.ndarray]) -> np.ndarray:
        processed_features = []
        
        for feat in features:
            feat = np.asarray(feat)            
            if feat.ndim > 1:
                feat = feat.flatten()
            
            processed_features.append(feat)
        
        feature_matrix = np.stack(processed_features)
        
        if feature_matrix.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got {feature_matrix.ndim}D")
            
        return feature_matrix
        
    def _plot_pca_2d(self, features: np.ndarray, video_paths: List[str], 
                     selected_indices: List[int], ax, query: str):
        if features.shape[0] < 2:
            ax.text(0.5, 0.5, 'Not enough samples for PCA', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('PCA Feature Space (Not enough samples)')
            return
            
        n_components = min(2, features.shape[1])
        pca = PCA(n_components=n_components)
        features_2d = pca.fit_transform(features)
        
        if n_components == 1:
            features_2d = np.column_stack([features_2d, np.zeros(features_2d.shape[0])])
        
        ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                  alpha=0.6, s=50, color='lightblue', label='All videos')
        
        if selected_indices:
            selected_features = features_2d[selected_indices]
            ax.scatter(selected_features[:, 0], selected_features[:, 1], 
                      alpha=0.8, s=100, color='red', label='Selected diverse', 
                      edgecolors='black', linewidth=1)
            
            for i, idx in enumerate(selected_indices):
                ax.annotate(f'{i+1}', 
                           (features_2d[idx, 0], features_2d[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
        
        if n_components >= 2:
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        else:
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel('Dummy dimension')
            
        ax.set_title('PCA Feature Space (2D Projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_distance_heatmap(
        self,
        features: np.ndarray,
        selected_indices: List[int],
        ax
    ):
        if not selected_indices:
            ax.text(0.5, 0.5, 'No selected samples', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Pairwise Cosine Distance Matrix')
            return

        features_subset = features[selected_indices]
        n = features_subset.shape[0]
        ax.set_title(f'Pairwise Cosine Distance Matrix ({n} selected videos)')

        distances = cosine_distances(features_subset)

        sns.heatmap(
            distances,
            annot=False,
            cmap='viridis',
            ax=ax,
            xticklabels=range(1, n+1),
            yticklabels=range(1, n+1),
            cbar_kws={'label': 'Cosine Distance'}
        )
        ax.set_xlabel('Selection Order')
        ax.set_ylabel('Selection Order')
        
    def _plot_feature_magnitudes(self, features: np.ndarray, 
                                selected_indices: List[int], ax):
        magnitudes = np.linalg.norm(features, axis=1)
        
        ax.hist(magnitudes, bins=20, alpha=0.7, color='skyblue', 
               label='All videos', edgecolor='black')
        
        if selected_indices:
            selected_mags = magnitudes[selected_indices]
            ax.hist(selected_mags, bins=20, alpha=0.8, color='red',
                   label='Selected diverse', edgecolor='black')
            
        ax.set_xlabel('Feature Vector Magnitude (L2 Norm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Magnitude Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_distance_distribution(self, features: np.ndarray, 
                                   selected_indices: List[int], ax):
        distances = cosine_distances(features)
        
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        all_distances = distances[upper_tri_indices]
        
        ax.hist(all_distances, bins=30, alpha=0.7, color='lightblue',
               label='All pairs', density=True, edgecolor='black')
        
        if selected_indices and len(selected_indices) > 1:
            selected_distances = []
            for i in range(len(selected_indices)):
                for j in range(i+1, len(selected_indices)):
                    idx1, idx2 = selected_indices[i], selected_indices[j]
                    selected_distances.append(distances[idx1, idx2])
            
            if selected_distances:
                ax.hist(selected_distances, bins=10, alpha=0.8, color='red',
                       label='Selected pairs', density=True, edgecolor='black')
                
                ax.axvline(np.mean(all_distances), color='blue', linestyle='--',
                          label=f'Mean all: {np.mean(all_distances):.3f}')
                ax.axvline(np.mean(selected_distances), color='red', linestyle='--',
                          label=f'Mean selected: {np.mean(selected_distances):.3f}')
        
        ax.set_xlabel('Cosine Distance')
        ax.set_ylabel('Density')
        ax.set_title('Pairwise Distance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def print_diversity_stats(self, features: List[np.ndarray], 
                             video_paths: List[str],
                             selected_indices: List[int]) -> None:
        if not features or not selected_indices:
            return
            
        feature_matrix = self._prepare_features(features)
        distances = cosine_distances(feature_matrix)
        
        upper_tri = np.triu_indices_from(distances, k=1)
        all_distances = distances[upper_tri]
        
        selected_distances = []
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                idx1, idx2 = selected_indices[i], selected_indices[j]
                selected_distances.append(distances[idx1, idx2])
        
        print("\n=== Diversity Statistics ===")
        print(f"Total videos: {len(features)}")
        print(f"Selected videos: {len(selected_indices)}")
        print(f"Feature dimension: {feature_matrix.shape[1]}")
        print(f"\nDistance Statistics:")
        print(f"  All pairs - Mean: {np.mean(all_distances):.4f}, Std: {np.std(all_distances):.4f}")
        if selected_distances:
            print(f"  Selected pairs - Mean: {np.mean(selected_distances):.4f}, Std: {np.std(selected_distances):.4f}")
            diversity_ratio = np.mean(selected_distances) / np.mean(all_distances)
            print(f"  Diversity ratio: {diversity_ratio:.4f}")
