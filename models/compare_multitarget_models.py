"""
Multi-Target Model Comparison: Predict col AND severity simultaneously
Compares Random Forest vs XGBoost, with and without PCA
Features are selected based on correlation analysis from notebook
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")


def load_data(data_path='data/model_ready.csv'):
    """Load preprocessed data."""
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def select_features_by_correlation(df, threshold=0.05):
    """
    Automatically select features based on correlation with both targets.
    Returns features with combined correlation score > threshold.
    """
    # All potential features
    all_features = ['lum', 'atm', 'agg', 'int', 'hour', 'day_of_week', 'month',
                    'num_users', 'num_killed', 'num_hospitalized', 'num_light_injury']
    
    # Filter to available features
    available_features = [f for f in all_features if f in df.columns]
    target_cols = ['col', 'max_severity']
    
    # Create correlation dataframe
    corr_df = df[available_features + target_cols].dropna()
    corr_matrix = corr_df.corr()
    
    # Calculate combined importance
    col_corr = corr_matrix['col'].drop(target_cols)
    sev_corr = corr_matrix['max_severity'].drop(target_cols)
    
    combined_scores = (np.abs(col_corr) + np.abs(sev_corr)) / 2
    
    # Select features above threshold
    selected = combined_scores[combined_scores > threshold].index.tolist()
    
    # Ensure minimum features
    if len(selected) < 3:
        print(f"   ⚠ Only {len(selected)} features above threshold {threshold}")
        print(f"   → Using top 5 features instead")
        selected = combined_scores.nlargest(5).index.tolist()
    
    return selected, combined_scores


def prepare_data(df):
    """Prepare features and targets for multi-target prediction."""
    print("\n" + "="*70)
    print("FEATURE SELECTION BASED ON CORRELATION")
    print("="*70)
    
    # Select features automatically
    selected_features, scores = select_features_by_correlation(df, threshold=0.05)
    
    print(f"\nSelected {len(selected_features)} features:")
    for feat in selected_features:
        print(f"  ✓ {feat:20s} (combined score: {scores[feat]:.4f})")
    
    # Prepare X and y
    X = df[selected_features].copy()
    y = df[['col', 'max_severity']].copy()
    
    # Handle missing values in features
    X = X.fillna(X.mode().iloc[0])
    
    # Remove rows with missing or invalid targets
    valid_mask = (
        y['col'].notna() & 
        y['max_severity'].notna() &
        (y['col'] > 0) &  # col should be 1-7
        (y['max_severity'] > 0)  # max_severity should be 1-4
    )
    
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    
    print(f"\nBefore conversion - col range: {y['col'].min()}-{y['col'].max()}, severity range: {y['max_severity'].min()}-{y['max_severity'].max()}")
    
    # Convert targets to 0-indexed for XGBoost compatibility
    # col: 1-7 -> 0-6, max_severity: 1-4 -> 0-3
    y['col'] = y['col'] - 1
    y['max_severity'] = y['max_severity'] - 1
    
    print(f"After conversion - col range: {y['col'].min()}-{y['col'].max()}, severity range: {y['max_severity'].min()}-{y['max_severity'].max()}")
    print(f"\nFinal dataset: {len(X):,} samples, {len(selected_features)} features")
    print(f"Targets: col (0-6 for 7 collision types), max_severity (0-3 for 4 severity levels)")
    
    return X, y, selected_features


def apply_pca(X_train, X_test, n_components=0.95):
    """Apply PCA to reduce dimensions while keeping 95% variance."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"   PCA: {X_train.shape[1]} features → {X_train_pca.shape[1]} components")
    print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_train_pca, X_test_pca, pca, scaler


def train_and_evaluate_multitarget(X_train, X_test, y_train, y_test, 
                                   model_type='rf', use_pca=False, save_model=False, selected_features=None):
    """Train and evaluate multi-target model."""
    print(f"\n{'='*70}")
    print(f"Training Multi-Target: {model_type.upper()} | PCA: {use_pca}")
    print(f"{'='*70}")
    
    # Apply PCA if requested
    pca_obj, scaler_obj = None, None
    if use_pca:
        X_train, X_test, pca_obj, scaler_obj = apply_pca(X_train, X_test)
    
    # Create base model
    if model_type == 'rf':
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgb':
        if not XGBOOST_AVAILABLE:
            print("   SKIPPED: XGBoost not available")
            return None, None
        base_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Wrap in MultiOutputClassifier for multi-target prediction
    model = MultiOutputClassifier(base_model)
    
    # Train
    print("   Training...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate each target separately
    metrics = {
        'model_type': model_type,
        'use_pca': use_pca,
        'n_features': X_train.shape[1]
    }
    
    target_names = ['collision', 'severity']
    for i, target_name in enumerate(target_names):
        y_true_target = y_test.iloc[:, i]
        y_pred_target = y_pred[:, i]
        
        acc = accuracy_score(y_true_target, y_pred_target)
        prec = precision_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
        rec = recall_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
        f1 = f1_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
        
        metrics[f'{target_name}_accuracy'] = acc
        metrics[f'{target_name}_precision'] = prec
        metrics[f'{target_name}_recall'] = rec
        metrics[f'{target_name}_f1'] = f1
        
        print(f"\n   {target_name.upper()}:")
        print(f"      Accuracy:  {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall:    {rec:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
    
    # Overall performance (average of both targets)
    metrics['avg_accuracy'] = (metrics['collision_accuracy'] + metrics['severity_accuracy']) / 2
    metrics['avg_f1'] = (metrics['collision_f1'] + metrics['severity_f1']) / 2
    
    print(f"\n   OVERALL AVERAGE:")
    print(f"      Accuracy:  {metrics['avg_accuracy']:.4f}")
    print(f"      F1-Score:  {metrics['avg_f1']:.4f}")
    
    # Save model if requested
    if save_model:
        model_name = f"{model_type}_{'pca' if use_pca else 'nopca'}_multitarget.pkl"
        model_path = Path("models") / model_name
        
        # Save model with metadata
        model_data = {
            'model': model,
            'pca': pca_obj,
            'scaler': scaler_obj,
            'features': selected_features,
            'use_pca': use_pca,
            'model_type': model_type,
            'metrics': metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n   ✓ Saved model: {model_name}")
    
    return metrics, model


def compare_all_models(X, y, selected_features):
    """Compare all 4 model configurations for multi-target prediction."""
    results = []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n" + "="*70)
    print("TRAINING ALL MODEL CONFIGURATIONS")
    print("="*70)
    
    # RF without PCA
    metrics, _ = train_and_evaluate_multitarget(
        X_train, X_test, y_train, y_test,
        model_type='rf', use_pca=False, save_model=True, selected_features=selected_features
    )
    if metrics:
        results.append(metrics)
    
    # RF with PCA
    metrics, _ = train_and_evaluate_multitarget(
        X_train, X_test, y_train, y_test,
        model_type='rf', use_pca=True, save_model=True, selected_features=selected_features
    )
    if metrics:
        results.append(metrics)
    
    # XGB without PCA
    metrics, _ = train_and_evaluate_multitarget(
        X_train, X_test, y_train, y_test,
        model_type='xgb', use_pca=False, save_model=True, selected_features=selected_features
    )
    if metrics:
        results.append(metrics)
    
    # XGB with PCA
    metrics, _ = train_and_evaluate_multitarget(
        X_train, X_test, y_train, y_test,
        model_type='xgb', use_pca=True, save_model=True, selected_features=selected_features
    )
    if metrics:
        results.append(metrics)
    
    return results


def visualize_comparison(results):
    """Create comprehensive comparison visualizations."""
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Collision accuracy
    ax = axes[0, 0]
    df_plot = df_results.set_index(['model_type', 'use_pca'])['collision_accuracy']
    df_plot.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Collision Type - Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model Configuration')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([f"{m.upper()}\n{'PCA' if p else 'No PCA'}" 
                        for m, p in df_plot.index], rotation=0)
    
    # 2. Collision F1-score
    ax = axes[0, 1]
    df_plot = df_results.set_index(['model_type', 'use_pca'])['collision_f1']
    df_plot.plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Collision Type - F1-Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('Model Configuration')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([f"{m.upper()}\n{'PCA' if p else 'No PCA'}" 
                        for m, p in df_plot.index], rotation=0)
    
    # 3. Severity accuracy
    ax = axes[0, 2]
    df_plot = df_results.set_index(['model_type', 'use_pca'])['severity_accuracy']
    df_plot.plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_title('Severity - Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model Configuration')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([f"{m.upper()}\n{'PCA' if p else 'No PCA'}" 
                        for m, p in df_plot.index], rotation=0)
    
    # 4. Severity F1-score
    ax = axes[1, 0]
    df_plot = df_results.set_index(['model_type', 'use_pca'])['severity_f1']
    df_plot.plot(kind='bar', ax=ax, color='orange')
    ax.set_title('Severity - F1-Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('Model Configuration')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([f"{m.upper()}\n{'PCA' if p else 'No PCA'}" 
                        for m, p in df_plot.index], rotation=0)
    
    # 5. Overall average performance
    ax = axes[1, 1]
    df_plot = df_results.set_index(['model_type', 'use_pca'])['avg_f1']
    df_plot.plot(kind='bar', ax=ax, color='purple')
    ax.set_title('Overall Average F1-Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average F1-Score')
    ax.set_xlabel('Model Configuration')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([f"{m.upper()}\n{'PCA' if p else 'No PCA'}" 
                        for m, p in df_plot.index], rotation=0)
    
    # 6. PCA impact comparison
    ax = axes[1, 2]
    pca_comparison = df_results.groupby('use_pca')['avg_f1'].mean()
    pca_comparison.plot(kind='bar', ax=ax, color=['red', 'green'])
    ax.set_title('PCA Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average F1-Score')
    ax.set_xlabel('PCA Usage')
    ax.set_xticklabels(['Without PCA', 'With PCA'], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/multitarget_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: models/multitarget_comparison.png")
    plt.show()


def print_summary(results):
    """Print comprehensive summary."""
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MULTI-TARGET PREDICTION - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    print("\nDetailed Results:")
    print("-"*80)
    for idx, row in df_results.iterrows():
        config = f"{row['model_type'].upper()} + {'PCA' if row['use_pca'] else 'No PCA'}"
        print(f"\n{config}:")
        print(f"  Features: {row['n_features']}")
        print(f"  Collision  - Acc: {row['collision_accuracy']:.4f}, F1: {row['collision_f1']:.4f}")
        print(f"  Severity   - Acc: {row['severity_accuracy']:.4f}, F1: {row['severity_f1']:.4f}")
        print(f"  AVERAGE    - Acc: {row['avg_accuracy']:.4f}, F1: {row['avg_f1']:.4f}")
    
    # Best overall model
    print("\n" + "="*80)
    print("BEST MODEL (by average F1-score)")
    print("="*80)
    best_idx = df_results['avg_f1'].idxmax()
    best = df_results.loc[best_idx]
    
    print(f"\nConfiguration: {best['model_type'].upper()} {'with' if best['use_pca'] else 'without'} PCA")
    print(f"Features: {best['n_features']}")
    print(f"\nCollision Type:")
    print(f"  Accuracy:  {best['collision_accuracy']:.4f}")
    print(f"  F1-Score:  {best['collision_f1']:.4f}")
    print(f"\nSeverity:")
    print(f"  Accuracy:  {best['severity_accuracy']:.4f}")
    print(f"  F1-Score:  {best['severity_f1']:.4f}")
    print(f"\nOverall Average:")
    print(f"  Accuracy:  {best['avg_accuracy']:.4f}")
    print(f"  F1-Score:  {best['avg_f1']:.4f}")
    
    # PCA analysis
    print("\n" + "="*80)
    print("PCA IMPACT ANALYSIS")
    print("="*80)
    
    with_pca = df_results[df_results['use_pca'] == True]['avg_f1'].mean()
    without_pca = df_results[df_results['use_pca'] == False]['avg_f1'].mean()
    diff = with_pca - without_pca
    
    print(f"\nAverage F1-Score:")
    print(f"  Without PCA: {without_pca:.4f}")
    print(f"  With PCA:    {with_pca:.4f}")
    print(f"  Difference:  {diff:+.4f}")
    
    if diff > 0.01:
        print("\n✓ PCA IMPROVES performance")
        print("  → Reduces features while maintaining/improving accuracy")
        print("  → Recommended for production")
    elif diff < -0.01:
        print("\n✗ PCA REDUCES performance")
        print("  → Original features work better")
        print("  → Keep all features for better results")
    else:
        print("\n≈ PCA has MINIMAL impact")
        print("  → Choice depends on other factors (speed, interpretability)")
    
    # Model type comparison
    print("\n" + "="*80)
    print("RANDOM FOREST vs XGBOOST")
    print("="*80)
    
    rf_score = df_results[df_results['model_type'] == 'rf']['avg_f1'].mean()
    xgb_score = df_results[df_results['model_type'] == 'xgb']['avg_f1'].mean()
    diff = xgb_score - rf_score
    
    print(f"\nAverage F1-Score:")
    print(f"  Random Forest: {rf_score:.4f}")
    print(f"  XGBoost:       {xgb_score:.4f}")
    print(f"  Difference:    {diff:+.4f}")
    
    if diff > 0.01:
        print("\n✓ XGBoost performs BETTER")
    elif diff < -0.01:
        print("\n✓ Random Forest performs BETTER")
    else:
        print("\n≈ Both models perform SIMILARLY")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print(f"\nUse: {best['model_type'].upper()} {'with' if best['use_pca'] else 'without'} PCA")
    print(f"This configuration achieves the best overall performance for")
    print(f"simultaneous prediction of collision type AND severity.")
    print("="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("MULTI-TARGET MODEL COMPARISON")
    print("Predicting col (collision) AND max_severity (severity) simultaneously")
    print("="*80)
    
    # Load data
    print("\n[Step 1] Loading data...")
    df = load_data()
    
    # Prepare features and targets
    print("\n[Step 2] Selecting features based on correlation...")
    X, y, selected_features = prepare_data(df)
    
    # Compare all models
    print("\n[Step 3] Training and comparing all models...")
    results = compare_all_models(X, y, selected_features)
    
    # Visualize
    print("\n[Step 4] Creating visualizations...")
    visualize_comparison(results)
    
    # Print summary
    print("\n[Step 5] Generating summary...")
    print_summary(results)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("\n✓ Saved 4 individual models:")
    print("  - models/rf_nopca_multitarget.pkl")
    print("  - models/rf_pca_multitarget.pkl")
    print("  - models/xgb_nopca_multitarget.pkl")
    print("  - models/xgb_pca_multitarget.pkl")
    print("\n✓ Dashboard can now use these models!")
    
    return results, selected_features


if __name__ == '__main__':
    main()
