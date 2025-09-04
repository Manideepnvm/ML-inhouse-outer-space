"""
Enhanced Visualization Module for Astronomical Object Classification
Provides comprehensive plotting and visualization tools with modern features and improved functionality.
"""

import numpy as np
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')

# Set enhanced styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

class EnhancedAstronomicalVisualizer:
    """
    Comprehensive and modern visualizer for astronomical data and ML results.
    """
    
    def __init__(self, style='modern'):
        # Modern color palettes
        self.colors = {
            'primary': ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'],
            'astronomical': ['#FFD700', '#4169E1', '#DC143C', '#32CD32', '#FF69B4'],
            'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c'],
            'cosmic': ['#1a1a2e', '#16213e', '#0f3460', '#533483']
        }
        
        # Astronomical object type mappings
        self.object_types = {
            0: 'Star', 1: 'Galaxy', 2: 'Quasar',
            'STAR': 'Star', 'GALAXY': 'Galaxy', 'QSO': 'Quasar'
        }
        
        # Initialize style
        self.set_style(style)
    
    def set_style(self, style='modern'):
        """
        Set the visualization style.
        
        Args:
            style (str): Style name ('modern', 'classic', 'minimal')
        """
        if style == 'modern':
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("Set2")
        elif style == 'classic':
            plt.style.use('classic')
            sns.set_palette("husl")
        elif style == 'minimal':
            plt.style.use('default')
            sns.set_palette("pastel")
        else:
            plt.style.use('default')
            sns.set_palette("Set2")
        
    def plot_enhanced_data_overview(self, data, target_col='class', figsize=(20, 15)):
        """
        Create comprehensive data overview with enhanced visualizations.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("🌌 Creating enhanced data overview...")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Target distribution with enhanced styling
        ax1 = fig.add_subplot(gs[0, 0])
        if target_col in data.columns:
            target_counts = data[target_col].value_counts()
            # Map to readable names
            labels = [self.object_types.get(idx, idx) for idx in target_counts.index]
            
            wedges, texts, autotexts = ax1.pie(target_counts.values, labels=labels, 
                                              autopct='%1.1f%%', colors=self.colors['astronomical'][:len(labels)],
                                              explode=[0.05] * len(labels), shadow=True, startangle=90)
            ax1.set_title('Astronomical Object Distribution', fontweight='bold', fontsize=12)
            
            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 2. Data quality heatmap
        ax2 = fig.add_subplot(gs[0, 1:3])
        missing_data = data.isnull().sum().sort_values(ascending=False)
        if missing_data.sum() > 0:
            missing_pct = (missing_data / len(data)) * 100
            bars = ax2.barh(range(len(missing_pct)), missing_pct.values,
                           color=['#E74C3C' if x > 5 else '#F39C12' if x > 0 else '#2ECC71' for x in missing_pct])
            ax2.set_yticks(range(len(missing_pct)))
            ax2.set_yticklabels(missing_pct.index)
            ax2.set_xlabel('Missing Values (%)')
            ax2.set_title('Data Quality Assessment', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        
        # 3. Statistical summary
        ax3 = fig.add_subplot(gs[0, 3])
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        stats_data = []
        for col in numerical_cols[:8]:  # Top 8 features
            stats_data.extend([data[col].mean(), data[col].std(), data[col].min(), data[col].max()])
        
        if stats_data:
            ax3.text(0.1, 0.9, f"Dataset Shape: {data.shape}", transform=ax3.transAxes, fontweight='bold')
            ax3.text(0.1, 0.8, f"Features: {len(numerical_cols)}", transform=ax3.transAxes)
            ax3.text(0.1, 0.7, f"Complete Records: {len(data.dropna())}", transform=ax3.transAxes)
            ax3.text(0.1, 0.6, f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB", 
                    transform=ax3.transAxes)
            ax3.set_title('Dataset Statistics', fontweight='bold')
            ax3.axis('off')
        
        # 4. Photometric bands visualization (SDSS specific)
        bands = ['u', 'g', 'r', 'i', 'z']
        available_bands = [band for band in bands if band in data.columns]
        
        if len(available_bands) >= 3:
            ax4 = fig.add_subplot(gs[1, :2])
            band_data = data[available_bands].mean()
            
            # Create stellar-like visualization
            angles = np.linspace(0, 2*np.pi, len(available_bands), endpoint=False)
            values = band_data.values
            
            # Close the plot
            angles = np.concatenate((angles, [angles[0]]))
            values = np.concatenate((values, [values[0]]))
            
            ax4.plot(angles, values, 'o-', linewidth=2, color='#3498DB')
            ax4.fill(angles, values, alpha=0.25, color='#3498DB')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(available_bands)
            ax4.set_title('Average Photometric Band Magnitudes', fontweight='bold')
            ax4.grid(True)
            
        # 5. Feature distributions by class
        ax5 = fig.add_subplot(gs[1, 2:])
        if target_col in data.columns and len(numerical_cols) > 0:
            feature = numerical_cols[0]  # Use first numerical feature
            classes = data[target_col].unique()
            
            for i, cls in enumerate(classes):
                subset = data[data[target_col] == cls][feature].dropna()
                label = self.object_types.get(cls, cls)
                ax5.hist(subset, bins=30, alpha=0.6, label=label, 
                        color=self.colors['astronomical'][i % len(self.colors['astronomical'])])
            
            ax5.set_xlabel(feature)
            ax5.set_ylabel('Frequency')
            ax5.set_title(f'{feature} Distribution by Object Type', fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        
        # 6. Color-color diagrams (astronomical specific)
        if all(band in data.columns for band in ['g', 'r', 'i']):
            ax6 = fig.add_subplot(gs[2, :2])
            
            # Calculate colors
            g_r = data['g'] - data['r']
            r_i = data['r'] - data['i']
            
            if target_col in data.columns:
                scatter = ax6.scatter(g_r, r_i, c=data[target_col].astype('category').cat.codes,
                                    cmap='viridis', alpha=0.6, s=20)
                ax6.set_xlabel('g - r (Color Index)')
                ax6.set_ylabel('r - i (Color Index)')
                ax6.set_title('Color-Color Diagram', fontweight='bold')
                ax6.grid(alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax6)
                cbar.set_label('Object Type')
        
        # 7. Redshift distribution (if available)
        if 'redshift' in data.columns:
            ax7 = fig.add_subplot(gs[2, 2:])
            
            # Log scale for redshift
            redshift_data = data['redshift'][data['redshift'] > 0]
            if len(redshift_data) > 0:
                ax7.hist(np.log10(redshift_data), bins=50, alpha=0.7, color='#9B59B6')
                ax7.set_xlabel('log₁₀(Redshift)')
                ax7.set_ylabel('Frequency')
                ax7.set_title('Redshift Distribution (log scale)', fontweight='bold')
                ax7.grid(alpha=0.3)
        
        plt.suptitle('🔭 Astronomical Data Comprehensive Overview', fontsize=20, fontweight='bold', y=0.98)
        plt.show()
    
    def plot_advanced_correlation_analysis(self, data, method='pearson', figsize=(16, 12)):
        """
        Advanced correlation analysis with multiple methods and clustering.
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            figsize (tuple): Figure size
        """
        print(f"🧬 Creating advanced correlation analysis ({method})...")
        
        numerical_data = data.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) < 2:
            print("⚠️ Need at least 2 numerical columns for correlation analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Advanced Correlation Analysis ({method.capitalize()})', fontsize=16, fontweight='bold')
        
        # 1. Enhanced correlation heatmap
        corr_matrix = numerical_data.corr(method=method)
        
        # Create custom mask for better visualization
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Enhanced heatmap with better styling
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0, 0],
                   fmt='.2f', annot_kws={'size': 8})
        axes[0, 0].set_title('Correlation Heatmap', fontweight='bold')
        
        # 2. Correlation network (top correlations)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.5:  # Threshold for significant correlation
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            # Sort by correlation strength
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = high_corr_pairs[:10]  # Top 10 correlations
            
            features = list(set([pair[0] for pair in top_pairs] + [pair[1] for pair in top_pairs]))
            y_pos = np.arange(len(top_pairs))
            corr_values = [pair[2] for pair in top_pairs]
            pair_labels = [f"{pair[0]} ↔ {pair[1]}" for pair in top_pairs]
            
            bars = axes[0, 1].barh(y_pos, corr_values, color=self.colors['gradient'][:len(top_pairs)])
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels(pair_labels, fontsize=8)
            axes[0, 1].set_xlabel('Correlation Strength')
            axes[0, 1].set_title('Top Feature Correlations', fontweight='bold')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Correlation clustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        distance_matrix = 1 - abs(corr_matrix)
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        dendrogram(linkage_matrix, labels=corr_matrix.columns, ax=axes[1, 0],
                  orientation='left', leaf_font_size=8)
        axes[1, 0].set_title('Feature Clustering (Correlation-based)', fontweight='bold')
        
        # 4. Correlation distribution
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        axes[1, 1].hist(corr_values, bins=30, alpha=0.7, color='#3498DB', edgecolor='black')
        axes[1, 1].axvline(np.mean(corr_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(corr_values):.3f}')
        axes[1, 1].set_xlabel('Correlation Coefficient')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Correlation Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_ml_performance_dashboard(self, results_dict, figsize=(20, 15)):
        """
        Comprehensive ML performance dashboard with multiple visualizations.
        
        Args:
            results_dict (dict): Dictionary of model results
            figsize (tuple): Figure size
        """
        print("🤖 Creating ML performance dashboard...")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Extract metrics for all models
        metrics_data = []
        for model_name, results in results_dict.items():
            if 'metrics' in results:
                metrics = results['metrics'].copy()
                metrics['Model'] = model_name
                metrics_data.append(metrics)
        
        if not metrics_data:
            print("⚠️ No metrics data found in results")
            return
            
        metrics_df = pd.DataFrame(metrics_data)
        
        # 1. Performance radar chart
        ax1 = fig.add_subplot(gs[0, :2], projection='polar')
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics_cols if m in metrics_df.columns]
        
        if len(available_metrics) >= 3:
            angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            for i, (_, row) in enumerate(metrics_df.iterrows()):
                values = [row[metric] for metric in available_metrics]
                values += [values[0]]  # Close the plot
                
                ax1.plot(angles, values, 'o-', linewidth=2, 
                        label=row['Model'], color=self.colors['primary'][i % len(self.colors['primary'])])
                ax1.fill(angles, values, alpha=0.1, color=self.colors['primary'][i % len(self.colors['primary'])])
            
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(available_metrics)
            ax1.set_title('Model Performance Radar', fontweight='bold', pad=20)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax1.grid(True)
        
        # 2. Confusion matrices comparison
        n_models = len(results_dict)
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'confusion_matrix' in results and i < 2:  # Show first 2 models
                ax = fig.add_subplot(gs[0, 2 + i])
                cm = results['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Star', 'Galaxy', 'Quasar'],
                           yticklabels=['Star', 'Galaxy', 'Quasar'], ax=ax)
                ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # 3. Metrics comparison bars
        ax3 = fig.add_subplot(gs[1, :])
        x = np.arange(len(metrics_df))
        width = 0.15
        
        for i, metric in enumerate(available_metrics):
            if metric in metrics_df.columns:
                offset = (i - len(available_metrics)/2) * width
                bars = ax3.bar(x + offset, metrics_df[metric], width, 
                              label=metric, color=self.colors['primary'][i % len(self.colors['primary'])],
                              alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_df['Model'], rotation=45)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 4. ROC curves
        ax4 = fig.add_subplot(gs[2, :2])
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'roc_data' in results:
                fpr, tpr, auc = results['roc_data']
                ax4.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC={auc:.3f})',
                        color=self.colors['primary'][i % len(self.colors['primary'])])
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Learning curves (if available)
        ax5 = fig.add_subplot(gs[2, 2:])
        learning_data_available = any('learning_curve' in results for results in results_dict.values())
        
        if learning_data_available:
            for i, (model_name, results) in enumerate(results_dict.items()):
                if 'learning_curve' in results:
                    train_scores, val_scores = results['learning_curve']
                    train_sizes = np.linspace(0.1, 1.0, len(train_scores))
                    
                    color = self.colors['primary'][i % len(self.colors['primary'])]
                    ax5.plot(train_sizes, train_scores, 'o-', color=color, 
                            label=f'{model_name} (Train)', alpha=0.8)
                    ax5.plot(train_sizes, val_scores, 's-', color=color, 
                            label=f'{model_name} (Val)', alpha=0.6, linestyle='--')
            
            ax5.set_xlabel('Training Set Size')
            ax5.set_ylabel('Accuracy Score')
            ax5.set_title('Learning Curves', fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        else:
            # Feature importance summary
            if any('feature_importance' in results for results in results_dict.values()):
                importance_data = []
                for model_name, results in results_dict.items():
                    if 'feature_importance' in results:
                        importance_data.extend(results['feature_importance'][:5])  # Top 5 features
                
                if importance_data:
                    feature_names = [item['feature'] for item in importance_data]
                    importance_values = [item['importance'] for item in importance_data]
                    
                    bars = ax5.bar(range(len(feature_names)), importance_values, 
                                  color=self.colors['gradient'][:len(feature_names)])
                    ax5.set_xticks(range(len(feature_names)))
                    ax5.set_xticklabels(feature_names, rotation=45, ha='right')
                    ax5.set_ylabel('Importance')
                    ax5.set_title('Top Feature Importance', fontweight='bold')
                    ax5.grid(axis='y', alpha=0.3)
        
        plt.suptitle('🤖 Machine Learning Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.show()
    
    def plot_dimensionality_reduction_suite(self, X, y=None, methods=['pca', 'tsne', 'umap'], figsize=(18, 12)):
        """
        Comprehensive dimensionality reduction visualization suite.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (array): Target variable (optional)
            methods (list): List of reduction methods
            figsize (tuple): Figure size
        """
        print("🔬 Creating dimensionality reduction suite...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_methods = len(methods)
        fig, axes = plt.subplots(2, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Dimensionality Reduction Analysis Suite', fontsize=16, fontweight='bold')
        
        for i, method in enumerate(methods):
            print(f"  🔹 Computing {method.upper()}...")
            
            if method.lower() == 'pca':
                # PCA analysis
                pca = PCA()
                X_reduced = pca.fit_transform(X_scaled)
                explained_var = pca.explained_variance_ratio_
                
                # Scree plot
                axes[0, i].plot(range(1, min(21, len(explained_var) + 1)), 
                               explained_var[:20], 'bo-', linewidth=2, markersize=6)
                axes[0, i].set_xlabel('Principal Component')
                axes[0, i].set_ylabel('Explained Variance Ratio')
                axes[0, i].set_title(f'PCA Scree Plot', fontweight='bold')
                axes[0, i].grid(alpha=0.3)
                
                # 2D projection
                if y is not None:
                    scatter = axes[1, i].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                               c=y, cmap='viridis', alpha=0.6, s=30)
                    axes[1, i].set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
                    axes[1, i].set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
                    axes[1, i].set_title('PCA 2D Projection', fontweight='bold')
                    plt.colorbar(scatter, ax=axes[1, i])
                
            elif method.lower() == 'tsne':
                # t-SNE analysis
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)//4))
                X_tsne = tsne.fit_transform(X_scaled)
                
                # Plot different perplexities comparison
                perplexities = [5, 15, 30, 50]
                available_perp = [p for p in perplexities if p < len(X_scaled)//3]
                
                if len(available_perp) > 1:
                    for j, perp in enumerate(available_perp[:4]):
                        tsne_temp = TSNE(n_components=2, random_state=42, perplexity=perp)
                        X_temp = tsne_temp.fit_transform(X_scaled[:1000])  # Subset for speed
                        
                        if j == 0:  # Show only first one due to space
                            if y is not None:
                                scatter = axes[0, i].scatter(X_temp[:, 0], X_temp[:, 1], 
                                                           c=y[:1000] if len(y) > 1000 else y, 
                                                           cmap='viridis', alpha=0.6, s=20)
                                axes[0, i].set_title(f't-SNE (perplexity={perp})', fontweight='bold')
                
                # Main t-SNE plot
                if y is not None:
                    scatter = axes[1, i].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                               c=y, cmap='viridis', alpha=0.6, s=30)
                    axes[1, i].set_xlabel('t-SNE 1')
                    axes[1, i].set_ylabel('t-SNE 2')
                    axes[1, i].set_title('t-SNE 2D Projection', fontweight='bold')
                    plt.colorbar(scatter, ax=axes[1, i])
            
            elif method.lower() == 'umap':
                try:
                    from umap import UMAP
                    # UMAP analysis
                    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
                    X_umap = umap_reducer.fit_transform(X_scaled)
                    
                    # Parameter comparison
                    n_neighbors_list = [5, 15, 30]
                    for j, n_neighbors in enumerate(n_neighbors_list[:2]):
                        umap_temp = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
                        X_temp = umap_temp.fit_transform(X_scaled)
                        
                        if j == 0:
                            if y is not None:
                                scatter = axes[0, i].scatter(X_temp[:, 0], X_temp[:, 1], 
                                                           c=y, cmap='viridis', alpha=0.6, s=20)
                                axes[0, i].set_title(f'UMAP (n_neighbors={n_neighbors})', fontweight='bold')
                    
                    # Main UMAP plot
                    if y is not None:
                        scatter = axes[1, i].scatter(X_umap[:, 0], X_umap[:, 1], 
                                                   c=y, cmap='viridis', alpha=0.6, s=30)
                        axes[1, i].set_xlabel('UMAP 1')
                        axes[1, i].set_ylabel('UMAP 2')
                        axes[1, i].set_title('UMAP 2D Projection', fontweight='bold')
                        plt.colorbar(scatter, ax=axes[1, i])
                        
                except ImportError:
                    axes[0, i].text(0.5, 0.5, 'UMAP not available\nInstall with:\npip install umap-learn', 
                                   ha='center', va='center', transform=axes[0, i].transAxes)
                    axes[1, i].text(0.5, 0.5, 'UMAP not available', 
                                   ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[0, i].set_title('UMAP (Not Available)', fontweight='bold')
                    axes[1, i].set_title('UMAP (Not Available)', fontweight='bold')
        
        plt.show()
    
    def create_interactive_3d_visualization(self, data, x_col, y_col, z_col, color_col=None, size_col=None):
        """
        Create interactive 3D scatter plot for astronomical data.
        
        Args:
            data (pd.DataFrame): Input dataset
            x_col, y_col, z_col (str): Column names for 3D coordinates
            color_col (str): Column for color mapping
            size_col (str): Column for size mapping
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D plot
        """
        print("🌌 Creating interactive 3D visualization...")
        
        # Prepare data
        plot_data = data[[x_col, y_col, z_col]].dropna()
        
        if color_col and color_col in data.columns:
            color_data = data.loc[plot_data.index, color_col]
            # Map to readable names if it's the target column
            if color_col == 'class':
                color_labels = [self.object_types.get(val, val) for val in color_data]
                hover_text = [f"Type: {label}<br>{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{z_col}: {z:.3f}" 
                             for label, x, y, z in zip(color_labels, plot_data[x_col], plot_data[y_col], plot_data[z_col])]
            else:
                hover_text = [f"{color_col}: {c}<br>{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{z_col}: {z:.3f}" 
                             for c, x, y, z in zip(color_data, plot_data[x_col], plot_data[y_col], plot_data[z_col])]
        else:
            color_data = None
            hover_text = [f"{x_col}: {x:.3f}<br>{y_col}: {y:.3f}<br>{z_col}: {z:.3f}" 
                         for x, y, z in zip(plot_data[x_col], plot_data[y_col], plot_data[z_col])]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        if color_data is not None:
            # Group by color for better legend
            unique_colors = np.unique(color_data)
            for color_val in unique_colors:
                mask = color_data == color_val
                subset_data = plot_data[mask]
                subset_hover = [hover_text[i] for i, m in enumerate(mask) if m]
                
                label = self.object_types.get(color_val, str(color_val))
                
                fig.add_trace(go.Scatter3d(
                    x=subset_data[x_col],
                    y=subset_data[y_col],
                    z=subset_data[z_col],
                    mode='markers',
                    name=label,
                    hovertemplate=subset_hover[0].split('<br>')[0] + '<br>' + 
                                 '<br>'.join(subset_hover[0].split('<br>')[1:]) + '<extra></extra>',
                    marker=dict(
                        size=6 if size_col is None else data.loc[subset_data.index, size_col] * 10,
                        opacity=0.7,
                        color=self.colors['astronomical'][list(unique_colors).index(color_val) % len(self.colors['astronomical'])]
                    )
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=plot_data[x_col],
                y=plot_data[y_col],
                z=plot_data[z_col],
                mode='markers',
                hovertext=hover_text,
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color='#3498DB'
                )
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Interactive 3D Astronomical Data Visualization',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                bgcolor='rgba(0,0,0,0.1)',
                xaxis=dict(gridcolor='rgba(128,128,128,0.3)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.3)'),
                zaxis=dict(gridcolor='rgba(128,128,128,0.3)')
            ),
            width=900,
            height=700,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_astronomical_hr_diagram(self, data, color_col='g', magnitude_col='r', 
                                   temperature_proxy='g-r', figsize=(12, 10)):
        """
        Create Hertzsprung-Russell diagram equivalent using photometric data.
        
        Args:
            data (pd.DataFrame): Input dataset
            color_col (str): Color column
            magnitude_col (str): Magnitude column
            temperature_proxy (str): Temperature proxy calculation
            figsize (tuple): Figure size
        """
        print("⭐ Creating astronomical H-R diagram...")
        
        # Check if required columns exist
        required_cols = ['g', 'r']
        if not all(col in data.columns for col in required_cols):
            print("⚠️ Need 'g' and 'r' band data for H-R diagram")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Astronomical Color-Magnitude Diagrams', fontsize=16, fontweight='bold')
        
        # Calculate color index
        color_index = data['g'] - data['r']
        magnitude = data[magnitude_col]
        
        # Remove invalid data
        valid_mask = ~(np.isnan(color_index) | np.isnan(magnitude))
        color_index = color_index[valid_mask]
        magnitude = magnitude[valid_mask]
        
        if 'class' in data.columns:
            target_data = data['class'][valid_mask]
            
            # Plot by object type
            for obj_type in np.unique(target_data):
                mask = target_data == obj_type
                label = self.object_types.get(obj_type, obj_type)
                color = self.colors['astronomical'][list(np.unique(target_data)).index(obj_type) % len(self.colors['astronomical'])]
                
                axes[0].scatter(color_index[mask], magnitude[mask], 
                               label=label, alpha=0.6, s=20, color=color)
            
            axes[0].set_xlabel('g - r (Color Index)')
            axes[0].set_ylabel(f'{magnitude_col} (Magnitude)')
            axes[0].set_title('Color-Magnitude Diagram by Type', fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[0].invert_yaxis()  # Brighter objects have lower magnitudes
        
        # Density plot
        axes[1].hexbin(color_index, magnitude, gridsize=30, cmap='plasma', alpha=0.8)
        axes[1].set_xlabel('g - r (Color Index)')
        axes[1].set_ylabel(f'{magnitude_col} (Magnitude)')
        axes[1].set_title('Density Distribution', fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_evolution_analysis(self, data, target_col='class', figsize=(16, 12)):
        """
        Analyze how features evolve across different object types.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("📈 Creating feature evolution analysis...")
        
        if target_col not in data.columns:
            print(f"⚠️ Target column '{target_col}' not found")
            return
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 3:
            print("⚠️ Need at least 3 numerical features for evolution analysis")
            return
        
        # Select top features for analysis
        top_features = numerical_cols[:8]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Evolution Across Astronomical Objects', fontsize=16, fontweight='bold')
        
        # 1. Box plots for feature distributions
        ax1 = axes[0, 0]
        feature_data = []
        feature_labels = []
        class_labels = []
        
        for feature in top_features[:4]:
            for class_val in data[target_col].unique():
                subset = data[data[target_col] == class_val][feature].dropna()
                feature_data.extend(subset.values)
                feature_labels.extend([feature] * len(subset))
                class_labels.extend([self.object_types.get(class_val, class_val)] * len(subset))
        
        if feature_data:
            df_box = pd.DataFrame({
                'Value': feature_data,
                'Feature': feature_labels,
                'Class': class_labels
            })
            
            sns.boxplot(data=df_box, x='Feature', y='Value', hue='Class', ax=ax1)
            ax1.set_title('Feature Distributions by Object Type', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(alpha=0.3)
        
        # 2. Parallel coordinates plot
        ax2 = axes[0, 1]
        if len(top_features) >= 4:
            subset_data = data[list(top_features[:4]) + [target_col]].dropna()
            
            for class_val in subset_data[target_col].unique():
                class_subset = subset_data[subset_data[target_col] == class_val]
                
                # Normalize features for better visualization
                normalized_features = StandardScaler().fit_transform(class_subset[top_features[:4]])
                
                # Plot mean trajectory
                mean_values = np.mean(normalized_features, axis=0)
                std_values = np.std(normalized_features, axis=0)
                
                label = self.object_types.get(class_val, class_val)
                color = self.colors['astronomical'][list(subset_data[target_col].unique()).index(class_val) % len(self.colors['astronomical'])]
                
                ax2.plot(range(len(mean_values)), mean_values, 'o-', 
                        label=label, linewidth=2, color=color, markersize=8)
                ax2.fill_between(range(len(mean_values)), 
                               mean_values - std_values, mean_values + std_values,
                               alpha=0.2, color=color)
            
            ax2.set_xticks(range(len(top_features[:4])))
            ax2.set_xticklabels(top_features[:4], rotation=45)
            ax2.set_ylabel('Normalized Value')
            ax2.set_title('Feature Profiles (Mean ± Std)', fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Feature importance heatmap
        ax3 = axes[1, 0]
        if len(top_features) >= 3:
            # Calculate feature statistics by class
            stats_matrix = []
            class_names = []
            
            for class_val in data[target_col].unique():
                class_subset = data[data[target_col] == class_val]
                stats = []
                for feature in top_features[:6]:
                    if feature in class_subset.columns:
                        # Use coefficient of variation as importance metric
                        mean_val = class_subset[feature].mean()
                        std_val = class_subset[feature].std()
                        cv = std_val / mean_val if mean_val != 0 else 0
                        stats.append(cv)
                
                if stats:
                    stats_matrix.append(stats)
                    class_names.append(self.object_types.get(class_val, class_val))
            
            if stats_matrix:
                sns.heatmap(stats_matrix, xticklabels=top_features[:len(stats_matrix[0])], 
                           yticklabels=class_names, annot=True, fmt='.2f', 
                           cmap='YlOrRd', ax=ax3)
                ax3.set_title('Feature Variability by Class\n(Coefficient of Variation)', fontweight='bold')
        
        # 4. Clustering analysis
        ax4 = axes[1, 1]
        if len(numerical_cols) >= 2:
            # K-means clustering on selected features
            feature_subset = data[top_features[:4]].dropna()
            
            if len(feature_subset) > 10:
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(feature_subset)
                
                # Find optimal number of clusters
                inertias = []
                k_range = range(2, min(11, len(feature_subset)//5))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(scaled_features)
                    inertias.append(kmeans.inertia_)
                
                ax4.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax4.set_xlabel('Number of Clusters (k)')
                ax4.set_ylabel('Within-cluster Sum of Squares')
                ax4.set_title('Elbow Method for Optimal Clusters', fontweight='bold')
                ax4.grid(alpha=0.3)
                
                # Highlight elbow point
                if len(inertias) >= 3:
                    # Simple elbow detection
                    diffs = np.diff(inertias)
                    second_diffs = np.diff(diffs)
                    if len(second_diffs) > 0:
                        elbow_idx = np.argmax(second_diffs) + 2
                        if elbow_idx < len(k_range):
                            ax4.axvline(k_range[elbow_idx], color='red', linestyle='--', 
                                       label=f'Suggested k={k_range[elbow_idx]}')
                            ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_model_comparison_dashboard(self, results_dict):
        """
        Create comprehensive interactive dashboard for model comparison.
        
        Args:
            results_dict (dict): Dictionary of model results
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        print("📊 Creating interactive model comparison dashboard...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Performance Metrics', 'ROC Curves', 'Precision-Recall', 
                           'Confusion Matrix', 'Feature Importance', 'Training Time'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Extract data
        models = list(results_dict.keys())
        colors = self.colors['primary'][:len(models)]
        
        # 1. Performance metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, model in enumerate(models):
            if 'metrics' in results_dict[model]:
                model_metrics = results_dict[model]['metrics']
                values = [model_metrics.get(metric, 0) for metric in metrics]
                
                fig.add_trace(
                    go.Bar(x=metrics, y=values, name=model, 
                          marker_color=colors[i % len(colors)],
                          opacity=0.8),
                    row=1, col=1
                )
        
        # 2. ROC Curves
        for i, (model, results) in enumerate(results_dict.items()):
            if 'roc_data' in results:
                fpr, tpr, auc = results['roc_data']
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines', 
                              name=f'{model} (AUC={auc:.3f})',
                              line=dict(color=colors[i % len(colors)], width=3)),
                    row=1, col=2
                )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                      line=dict(dash='dash', color='gray'),
                      name='Random', showlegend=False),
            row=1, col=2
        )
        
        # 3. Precision-Recall curves
        for i, (model, results) in enumerate(results_dict.items()):
            if 'pr_data' in results:
                precision, recall, ap = results['pr_data']
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, mode='lines',
                              name=f'{model} (AP={ap:.3f})',
                              line=dict(color=colors[i % len(colors)], width=3)),
                    row=2, col=1
                )
        
        # 4. Confusion matrix (first model as example)
        if models and 'confusion_matrix' in results_dict[models[0]]:
            cm = results_dict[models[0]]['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, x=['Star', 'Galaxy', 'Quasar'], 
                          y=['Star', 'Galaxy', 'Quasar'],
                          colorscale='Blues', showscale=False),
                row=2, col=2
            )
        
        # 5. Feature importance comparison
        importance_data = {}
        for model, results in results_dict.items():
            if 'feature_importance' in results:
                for item in results['feature_importance'][:5]:  # Top 5 features
                    feature = item['feature']
                    if feature not in importance_data:
                        importance_data[feature] = {}
                    importance_data[feature][model] = item['importance']
        
        if importance_data:
            features = list(importance_data.keys())
            for i, model in enumerate(models):
                if any(model in importance_data[f] for f in features):
                    values = [importance_data[f].get(model, 0) for f in features]
                    fig.add_trace(
                        go.Bar(x=features, y=values, name=model,
                              marker_color=colors[i % len(colors)],
                              opacity=0.8),
                        row=3, col=1
                    )
        
        # 6. Training time comparison
        training_times = []
        model_names = []
        for model, results in results_dict.items():
            if 'training_time' in results:
                training_times.append(results['training_time'])
                model_names.append(model)
        
        if training_times:
            fig.add_trace(
                go.Bar(x=model_names, y=training_times,
                      marker_color=colors[:len(training_times)],
                      opacity=0.8, showlegend=False),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="🤖 ML Model Comparison Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Metrics", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Recall", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=1)
        fig.update_xaxes(title_text="Features", row=3, col=1)
        fig.update_yaxes(title_text="Importance", row=3, col=1)
        fig.update_xaxes(title_text="Models", row=3, col=2)
        fig.update_yaxes(title_text="Training Time (s)", row=3, col=2)
        
        return fig
    
    def plot_advanced_feature_analysis(self, data, target_col='class', figsize=(18, 14)):
        """
        Advanced feature analysis with statistical insights.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("🔍 Creating advanced feature analysis...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            print("⚠️ Need at least 2 numerical features for analysis")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Feature distribution comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        # Select top 6 features for visualization
        top_features = numerical_cols[:6]
        n_features = len(top_features)
        
        # Create violin plots
        violin_data = []
        for feature in top_features:
            for class_val in data[target_col].unique():
                subset = data[data[target_col] == class_val][feature].dropna()
                for val in subset:
                    violin_data.append({
                        'Feature': feature,
                        'Value': val,
                        'Class': self.object_types.get(class_val, class_val)
                    })
        
        if violin_data:
            violin_df = pd.DataFrame(violin_data)
            sns.violinplot(data=violin_df, x='Feature', y='Value', hue='Class', 
                          ax=ax1, palette=self.colors['astronomical'])
            ax1.set_title('Feature Value Distributions by Object Type', fontweight='bold', fontsize=14)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(alpha=0.3)
        
        # 2. Feature separability analysis
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate separability scores (using ANOVA F-statistic)
        from sklearn.feature_selection import f_classif
        
        X = data[numerical_cols].fillna(data[numerical_cols].mean())
        y = data[target_col]
        
        f_scores, p_values = f_classif(X, y)
        separability_df = pd.DataFrame({
            'Feature': numerical_cols,
            'F_Score': f_scores,
            'P_Value': p_values
        }).sort_values('F_Score', ascending=True)
        
        # Plot top features
        top_sep_features = separability_df.tail(10)
        bars = ax2.barh(range(len(top_sep_features)), top_sep_features['F_Score'],
                       color=self.colors['gradient'][:len(top_sep_features)])
        ax2.set_yticks(range(len(top_sep_features)))
        ax2.set_yticklabels(top_sep_features['Feature'], fontsize=9)
        ax2.set_xlabel('F-Score (Separability)')
        ax2.set_title('Feature Separability Analysis', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Feature correlation with target
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate correlation with encoded target
        if target_col in data.columns:
            target_encoded = data[target_col].astype('category').cat.codes
            correlations = []
            
            for col in numerical_cols:
                corr = np.corrcoef(data[col].fillna(data[col].mean()), target_encoded)[0, 1]
                correlations.append(abs(corr))
            
            corr_df = pd.DataFrame({
                'Feature': numerical_cols,
                'Correlation': correlations
            }).sort_values('Correlation', ascending=True)
            
            # Plot
            bars = ax3.barh(range(len(corr_df.tail(10))), corr_df.tail(10)['Correlation'],
                           color=[self.colors['primary'][i % len(self.colors['primary'])] 
                                 for i in range(len(corr_df.tail(10)))])
            ax3.set_yticks(range(len(corr_df.tail(10))))
            ax3.set_yticklabels(corr_df.tail(10)['Feature'], fontsize=9)
            ax3.set_xlabel('|Correlation with Target|')
            ax3.set_title('Target Correlation Strength', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. Feature interactions heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Calculate feature interactions (mutual information approximation)
        interaction_matrix = np.zeros((len(top_features[:6]), len(top_features[:6])))
        feature_subset = top_features[:6]
        
        for i, feat1 in enumerate(feature_subset):
            for j, feat2 in enumerate(feature_subset):
                if i != j and feat1 in data.columns and feat2 in data.columns:
                    # Simple interaction: correlation of products
                    interaction = abs(np.corrcoef(data[feat1] * data[feat2], target_encoded)[0, 1])
                    interaction_matrix[i, j] = interaction
        
        sns.heatmap(interaction_matrix, xticklabels=feature_subset, yticklabels=feature_subset,
                   annot=True, fmt='.2f', cmap='viridis', ax=ax4)
        ax4.set_title('Feature Interactions\n(Product Correlations)', fontweight='bold')
        
        # 5. Outlier detection visualization
        ax5 = fig.add_subplot(gs[2, :])
        
        # Use IQR method for outlier detection
        outlier_counts = {}
        for feature in top_features[:8]:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_counts[feature] = len(outliers)
        
        if outlier_counts:
            features = list(outlier_counts.keys())
            counts = list(outlier_counts.values())
            
            bars = ax5.bar(features, counts, color=self.colors['gradient'][:len(features)], alpha=0.8)
            ax5.set_xlabel('Features')
            ax5.set_ylabel('Number of Outliers')
            ax5.set_title('Outlier Detection Summary (IQR Method)', fontweight='bold', fontsize=14)
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('🔬 Advanced Feature Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.show()
    
    def create_astronomical_catalog_explorer(self, data, ra_col='ra', dec_col='dec', 
                                           magnitude_col='r', target_col='class'):
        """
        Create interactive astronomical catalog explorer.
        
        Args:
            data (pd.DataFrame): Input dataset
            ra_col (str): Right ascension column
            dec_col (str): Declination column  
            magnitude_col (str): Magnitude column
            target_col (str): Target column
            
        Returns:
            plotly.graph_objects.Figure: Interactive explorer
        """
        print("🌟 Creating astronomical catalog explorer...")
        
        # Check required columns
        required_cols = [ra_col, dec_col]
        if not all(col in data.columns for col in required_cols):
            print(f"⚠️ Required columns {required_cols} not found")
            return None 

        # Create scatter plot
        fig = go.Figure(data=go.Scattergeo(
            lon=data[ra_col],
            lat=data[dec_col],
            text=data[magnitude_col],
            mode='markers',
            marker=dict(
                size=5,
                color=data[target_col],
                colorscale='Viridis',
                colorbar=dict(title=target_col)
            )
        ))

        fig.update_layout(
            title='Astronomical Catalog Explorer',
            geo=dict(
                scope='world',
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                subunitcolor='rgb(217, 217, 217)',
                countrycolor='rgb(217, 217, 217)',
            )
        )

        return fig
    
    def plot_data_distribution(self, data, target_col='class', figsize=(12, 8)):
        """
        Plot data distribution for the target variable.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("📊 Creating data distribution plot...")
        
        if target_col not in data.columns:
            print(f"⚠️ Target column '{target_col}' not found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Class distribution
        class_counts = data[target_col].value_counts()
        labels = [self.object_types.get(idx, idx) for idx in class_counts.index]
        
        axes[0].pie(class_counts.values, labels=labels, autopct='%1.1f%%', 
                   colors=self.colors['astronomical'][:len(labels)], startangle=90)
        axes[0].set_title('Class Distribution')
        
        # Class counts bar plot
        bars = axes[1].bar(labels, class_counts.values, 
                          color=self.colors['astronomical'][:len(labels)])
        axes[1].set_title('Class Counts')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data, figsize=(12, 10)):
        """
        Plot correlation matrix for numerical features.
        
        Args:
            data (pd.DataFrame): Input dataset
            figsize (tuple): Figure size
        """
        print("📊 Creating correlation matrix...")
        
        numerical_data = data.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) < 2:
            print("⚠️ Need at least 2 numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f', annot_kws={'size': 8})
        
        plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance from a DataFrame.
        
        Args:
            importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        print(f"🎯 Creating feature importance plot (top {top_n})...")
        
        if importance_df is None or len(importance_df) == 0:
            print("⚠️ No feature importance data provided")
            return
        
        # Sort by importance and take top N
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors['gradient'][:len(top_features)])
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_dimensionality_reduction(self, X, y=None, method='pca', figsize=(12, 8)):
        """
        Plot dimensionality reduction results.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target variable (optional)
            method (str): Reduction method ('pca', 'tsne', 'umap')
            figsize (tuple): Figure size
        """
        print(f"🔍 Creating {method.upper()} dimensionality reduction plot...")
        
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)
            title = f'PCA (Explained Variance: {reducer.explained_variance_ratio_.sum():.2%})'
            
        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)//4))
            X_reduced = reducer.fit_transform(X_scaled)
            title = 't-SNE Visualization'
            
        elif method.lower() == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
                title = 'UMAP Visualization'
            except ImportError:
                print("⚠️ UMAP not available. Using PCA instead.")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                X_reduced = reducer.fit_transform(X_scaled)
                title = 'PCA (UMAP not available)'
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if y is not None:
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                                cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=30)
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(title, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_astronomical_features(self, data, target_col='class', figsize=(15, 10)):
        """
        Plot astronomical-specific features.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            figsize (tuple): Figure size
        """
        print("🌟 Creating astronomical features analysis...")
        
        # Check for photometric bands
        bands = ['u', 'g', 'r', 'i', 'z']
        available_bands = [band for band in bands if band in data.columns]
        
        if len(available_bands) < 2:
            print("⚠️ Need at least 2 photometric bands for astronomical analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Astronomical Features Analysis', fontsize=16, fontweight='bold')
        
        # 1. Color-color diagram
        if 'g' in data.columns and 'r' in data.columns and 'i' in data.columns:
            g_r = data['g'] - data['r']
            r_i = data['r'] - data['i']
            
            if target_col in data.columns:
                scatter = axes[0, 0].scatter(g_r, r_i, c=data[target_col].astype('category').cat.codes,
                                           cmap='viridis', alpha=0.6, s=20)
                axes[0, 0].set_xlabel('g - r (Color Index)')
                axes[0, 0].set_ylabel('r - i (Color Index)')
                axes[0, 0].set_title('Color-Color Diagram')
                axes[0, 0].grid(alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 0], label='Object Type')
        
        # 2. Magnitude distribution
        if 'r' in data.columns:
            if target_col in data.columns:
                for obj_type in data[target_col].unique():
                    subset = data[data[target_col] == obj_type]['r'].dropna()
                    label = self.object_types.get(obj_type, obj_type)
                    axes[0, 1].hist(subset, bins=30, alpha=0.6, label=label)
                
                axes[0, 1].set_xlabel('r-band Magnitude')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Magnitude Distribution by Type')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
        
        # 3. Redshift distribution
        if 'redshift' in data.columns:
            redshift_data = data['redshift'][data['redshift'] > 0]
            if len(redshift_data) > 0:
                axes[1, 0].hist(np.log10(redshift_data), bins=50, alpha=0.7, color='#9B59B6')
                axes[1, 0].set_xlabel('log₁₀(Redshift)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Redshift Distribution (log scale)')
                axes[1, 0].grid(alpha=0.3)
        
        # 4. Sky coordinates
        if 'ra' in data.columns and 'dec' in data.columns:
            if target_col in data.columns:
                scatter = axes[1, 1].scatter(data['ra'], data['dec'], 
                                           c=data[target_col].astype('category').cat.codes,
                                           cmap='viridis', alpha=0.6, s=10)
                axes[1, 1].set_xlabel('Right Ascension (degrees)')
                axes[1, 1].set_ylabel('Declination (degrees)')
                axes[1, 1].set_title('Sky Distribution')
                axes[1, 1].grid(alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1], label='Object Type')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, data, target_col='class'):
        """
        Create an interactive dashboard for the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        print("📊 Creating interactive dashboard...")
        
        try:
            # Create a simple interactive scatter plot
            fig = px.scatter(data, x='ra', y='dec', color=target_col,
                           title='Interactive Astronomical Data Explorer',
                           labels={'ra': 'Right Ascension', 'dec': 'Declination'})
            
            fig.update_layout(
                width=1000,
                height=600,
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            print(f"⚠️ Could not create interactive dashboard: {e}")
            return None