"""
machine.py

This module conducts insightful machine learning analyses for flight delay prediction.
Includes validation, feature importance analysis, and model comparisons.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, Markdown
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')


# Utility Functions
def display_title(s, pref='Figure', num=1):
    """Display formatted title for analysis sections."""
    s = f'<p><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></p>'
    display(Markdown(s))


def display_result(title, content):
    """Display formatted results."""
    display(Markdown(f"**{title}**\n\n{content}"))


#--------------------------------
# Analysis 1: Feature Engineering & Importance
def analyze_feature_importance(df, n_top_features=10):
    """
    INSIGHTFUL ANALYSIS: Engineers temporal and interaction features, then uses
    Random Forest to identify which features are most predictive of arrival delays.
    
    Insight: Understanding which features drive delays helps airlines focus
    operational improvements where they matter most.
    
    Parameters:
    -----------
    n_top_features : int
        Number of top features to display
        
    Returns:
    dict : Results including feature importances and model performance
    """
    display_title('Feature Engineering & Importance Analysis', num=5)
    
    # Prepare data
    df_clean = df.dropna(subset=['arrival_delay', 'departure_delay', 
                                  'distance', 'scheduled_dep_time']).copy()
    
    # Engineer features
    df_clean['dep_hour'] = df_clean['scheduled_dep_time'] // 100
    df_clean['dep_minute'] = df_clean['scheduled_dep_time'] % 100
    df_clean['time_of_day'] = pd.cut(df_clean['dep_hour'], 
                                      bins=[0, 6, 12, 18, 24],
                                      labels=['night', 'morning', 'afternoon', 'evening'])
    
    # Interaction features
    df_clean['delay_per_mile'] = df_clean['departure_delay'] / (df_clean['distance'] + 1)
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
    df_clean['is_peak_hour'] = df_clean['dep_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Seasonal features
    df_clean['is_winter'] = df_clean['month'].isin([12, 1, 2]).astype(int)
    df_clean['is_summer'] = df_clean['month'].isin([6, 7, 8]).astype(int)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_clean, columns=['time_of_day', 'airline', 'destination'], 
                                 drop_first=True)
    
    # Prepare features and target
    exclude_cols = ['arrival_delay', 'scheduled_dep_time']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols].values
    y = df_encoded['arrival_delay'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                     min_samples_split=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Feature importance
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top features
    top_features = feature_importance_df.head(n_top_features)
    axs[0].barh(range(len(top_features)), top_features['importance'], color='steelblue')
    axs[0].set_yticks(range(len(top_features)))
    axs[0].set_yticklabels(top_features['feature'])
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Importance', fontsize=11)
    axs[0].set_title(f'Top {n_top_features} Features by Importance', fontsize=12)
    axs[0].grid(alpha=0.3, axis='x')
    
    # Predicted vs Actual
    axs[1].scatter(y_test, y_pred_test, alpha=0.3, s=10, color='coral')
    axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axs[1].set_xlabel('Actual Arrival Delay (min)', fontsize=11)
    axs[1].set_ylabel('Predicted Arrival Delay (min)', fontsize=11)
    axs[1].set_title(f'Test Set: R² = {test_r2:.3f}, RMSE = {test_rmse:.2f}', fontsize=12)
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results
    result_text = f"""
- **Training R²**: {train_r2:.4f}, **RMSE**: {train_rmse:.2f} min
- **Test R²**: {test_r2:.4f}, **RMSE**: {test_rmse:.2f} min
- **Top 3 Features**: {', '.join(top_features['feature'].head(3).tolist())}
- **Interpretation**: {'Departure delay dominates predictions, but temporal and distance features add predictive value. Model shows good generalization with minimal overfitting.' if abs(train_r2 - test_r2) < 0.1 else 'Some overfitting detected; regularization may help.'}
"""
    display_result("Random Forest Performance & Feature Importance", result_text)
    
    return {
        'model': rf_model,
        'feature_importance': feature_importance_df,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'feature_cols': feature_cols
    }


#--------------------------------
# Analysis 2: Model Comparison with Cross-Validation
def compare_models_with_cv(df, cv_folds=5):
    """
    INSIGHTFUL ANALYSIS: Compares multiple regression models using cross-validation
    to identify the best approach for delay prediction.
    
    Models compared: Linear (Ridge/Lasso), Tree-based (Random Forest, Gradient Boosting)
    
    Insight: Different models capture different aspects of the delay phenomenon.
    Ensemble methods typically perform best but require more computation.
    
    Parameters:
    -----------
    cv_folds : int
        Number of cross-validation folds

    """
    display_title('Model Comparison with Cross-Validation', num=4)
    
    # Prepare data with key features only (to speed up computation)
    df_clean = df.dropna(subset=['arrival_delay', 'departure_delay', 
                                  'distance', 'day_of_week', 'month']).copy()
    
    # Basic feature engineering
    df_clean['dep_hour'] = df_clean['scheduled_dep_time'] // 100
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
    df_clean['is_peak_hour'] = df_clean['dep_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Select features
    feature_cols = ['departure_delay', 'distance', 'day_of_week', 'month', 
                    'dep_hour', 'is_weekend', 'is_peak_hour']
    
    X = df_clean[feature_cols].values
    y = df_clean['arrival_delay'].values
    
    # Standardize features (important for Ridge/Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                         test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=50, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, 
                                               min_samples_split=50, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, 
                                                       learning_rate=0.1, random_state=42)
    }
    
    # Cross-validation
    cv_results = {}
    test_results = {}
    
    for name, model in models.items():
        # Cross-validation scores (negative MSE, convert to RMSE)
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)
        
        # Train on full training set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        cv_results[name] = {
            'cv_mean': cv_rmse.mean(),
            'cv_std': cv_rmse.std()
        }
        test_results[name] = {
            'rmse': test_rmse,
            'r2': test_r2,
            'mae': test_mae
        }
        
        print(f"{name}: CV RMSE = {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # CV results
    model_names = list(cv_results.keys())
    cv_means = [cv_results[m]['cv_mean'] for m in model_names]
    cv_stds = [cv_results[m]['cv_std'] for m in model_names]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    axs[0].bar(range(len(model_names)), cv_means, yerr=cv_stds, 
               color=colors, alpha=0.7, capsize=5)
    axs[0].set_xticks(range(len(model_names)))
    axs[0].set_xticklabels(model_names, rotation=45, ha='right')
    axs[0].set_ylabel('Cross-Validation RMSE (min)', fontsize=11)
    axs[0].set_title(f'{cv_folds}-Fold Cross-Validation Results', fontsize=12)
    axs[0].grid(alpha=0.3, axis='y')
    
    # Test set comparison
    metrics = ['RMSE', 'MAE', 'R²']
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(['rmse', 'mae', 'r2']):
        values = [test_results[m][metric] for m in model_names]
        # Normalize R² to similar scale for visualization
        if metric == 'r2':
            values = [v * 100 for v in values]  # Scale R² to percentage
        axs[1].bar(x_pos + i*width, values, width, label=metrics[i], alpha=0.7)
    
    axs[1].set_xticks(x_pos + width)
    axs[1].set_xticklabels(model_names, rotation=45, ha='right')
    axs[1].set_ylabel('Metric Value', fontsize=11)
    axs[1].set_title('Test Set Performance Comparison', fontsize=12)
    axs[1].legend()
    axs[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = min(test_results.keys(), key=lambda k: test_results[k]['rmse'])
    best_rmse = test_results[best_model_name]['rmse']
    best_r2 = test_results[best_model_name]['r2']
    
    result_text = f"""
- **Best Model**: {best_model_name} (Test RMSE: {best_rmse:.2f} min, R²: {best_r2:.4f})
- **Cross-Validation**: Provides robust performance estimates across {cv_folds} folds
- **Key Finding**: {'Ensemble methods (RF, GB) outperform linear models, capturing non-linear delay patterns.' if best_model_name in ['Random Forest', 'Gradient Boosting'] else 'Simpler models perform competitively, suggesting linear relationships dominate.'}
- **Practical Implication**: {best_model_name} offers the best balance of accuracy and interpretability for operational use.
"""
    display_result("Model Comparison Summary", result_text)
    
    return {
        'cv_results': cv_results,
        'test_results': test_results,
        'best_model': best_model_name,
        'models': models
    }


#--------------------------------
# Analysis 3: Learning Curves & Overfitting Analysis

def analyze_learning_curves(df, model_type='RandomForest'):
    """
    INSIGHTFUL ANALYSIS: Generates learning curves to understand how model
    performance changes with training set size.
    
    Key insight: Identifies whether models benefit from more data or are
    limited by model complexity. Helps diagnose overfitting/underfitting.
    

    """
    display_title('Learning Curves & Overfitting Analysis', num=3)
    
    # Prepare data
    df_clean = df.dropna(subset=['arrival_delay', 'departure_delay', 
                                  'distance', 'day_of_week']).copy()
    
    df_clean['dep_hour'] = df_clean['scheduled_dep_time'] // 100
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
    
    feature_cols = ['departure_delay', 'distance', 'day_of_week', 'dep_hour', 'is_weekend']
    X = df_clean[feature_cols].values
    y = df_clean['arrival_delay'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Defining the model
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    # Generate learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []
    
    print(f"Generating learning curves for {model_type}...")
    for size in train_sizes:
        n_samples = int(size * len(X_train))
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        model.fit(X_subset, y_subset)
        
        train_pred = model.predict(X_subset)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_subset, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_scores.append(train_rmse)
        test_scores.append(test_rmse)
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Learning curves
    train_samples = (train_sizes * len(X_train)).astype(int)
    axs[0].plot(train_samples, train_scores, 'o-', color='blue', label='Training RMSE')
    axs[0].plot(train_samples, test_scores, 'o-', color='red', label='Test RMSE')
    axs[0].fill_between(train_samples, train_scores, test_scores, alpha=0.1, color='gray')
    axs[0].set_xlabel('Training Set Size', fontsize=11)
    axs[0].set_ylabel('RMSE (min)', fontsize=11)
    axs[0].set_title(f'Learning Curves: {model_type}', fontsize=12)
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    # Overfitting gap
    gap = np.array(test_scores) - np.array(train_scores)
    axs[1].plot(train_samples, gap, 'o-', color='purple', linewidth=2)
    axs[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axs[1].set_xlabel('Training Set Size', fontsize=11)
    axs[1].set_ylabel('Overfitting Gap\n(Test RMSE - Train RMSE)', fontsize=11)
    axs[1].set_title('Overfitting Analysis', fontsize=12)
    axs[1].grid(alpha=0.3)
    axs[1].fill_between(train_samples, 0, gap, where=(gap > 0), 
                        alpha=0.3, color='red', label='Overfitting Region')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    final_gap = test_scores[-1] - train_scores[-1]
    converged = abs(test_scores[-1] - test_scores[-3]) < 1.0
    
    result_text = f"""
- **Final Training RMSE**: {train_scores[-1]:.2f} min
- **Final Test RMSE**: {test_scores[-1]:.2f} min
- **Overfitting Gap**: {final_gap:.2f} min
- **Convergence**: {'Model has converged; more data unlikely to help significantly.' if converged else 'Model still improving; more data could reduce test error.'}
- **Recommendation**: {'Current model complexity is appropriate.' if final_gap < 5 else 'Consider regularization to reduce overfitting.' if final_gap > 10 else 'Model shows healthy generalization.'}
"""
    display_result("Learning Curve Analysis", result_text)
    
    return {
        'train_sizes': train_samples,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'overfitting_gap': final_gap
    }



#--------------------------------
# Analysis 4: Residual Analysis & Error Patterns
def analyze_residuals(df):
    """
    INSIGHTFUL ANALYSIS: Examines prediction errors to identify systematic
    patterns that the model fails to capture.
    
    Insight: Understanding where models fail guides feature engineering
    and reveals operational insights about delay patterns.

    """
    display_title('Residual Analysis & Error Patterns', num=4)
    
    # Prepare data
    df_clean = df.dropna(subset=['arrival_delay', 'departure_delay', 
                                  'distance', 'day_of_week', 'month']).copy()
    
    df_clean['dep_hour'] = df_clean['scheduled_dep_time'] // 100
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([6, 7]).astype(int)
    
    feature_cols = ['departure_delay', 'distance', 'day_of_week', 
                    'month', 'dep_hour', 'is_weekend']
    X = df_clean[feature_cols].values
    y = df_clean['arrival_delay'].values
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions and residuals
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    # Attach residuals to test data
    test_indices = df_clean.index[len(X_train):]
    df_test = df_clean.loc[test_indices].copy()
    df_test['residual'] = residuals
    df_test['predicted'] = y_pred
    
    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Predicted
    axs[0, 0].scatter(y_pred, residuals, alpha=0.3, s=10, color='blue')
    axs[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axs[0, 0].set_xlabel('Predicted Arrival Delay (min)', fontsize=11)
    axs[0, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axs[0, 0].set_title('Residual Plot', fontsize=12)
    axs[0, 0].grid(alpha=0.3)
    
    # 2. Histogram of residuals
    axs[0, 1].hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
    axs[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axs[0, 1].set_xlabel('Residuals (min)', fontsize=11)
    axs[0, 1].set_ylabel('Frequency', fontsize=11)
    axs[0, 1].set_title(f'Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})', 
                        fontsize=12)
    axs[0, 1].grid(alpha=0.3, axis='y')
    
    # 3. Residuals by departure hour
    hour_residuals = df_test.groupby('dep_hour')['residual'].agg(['mean', 'std', 'count'])
    hour_residuals = hour_residuals[hour_residuals['count'] > 50]  # Filter low counts
    
    axs[1, 0].errorbar(hour_residuals.index, hour_residuals['mean'], 
                       yerr=hour_residuals['std'], fmt='o-', color='orange', capsize=3)
    axs[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axs[1, 0].set_xlabel('Departure Hour', fontsize=11)
    axs[1, 0].set_ylabel('Mean Residual (min)', fontsize=11)
    axs[1, 0].set_title('Prediction Bias by Time of Day', fontsize=12)
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].set_xticks(range(0, 24, 3))
    
    # 4. Residuals by distance bins
    df_test['distance_bin'] = pd.cut(df_test['distance'], bins=5)
    distance_residuals = df_test.groupby('distance_bin')['residual'].agg(['mean', 'std'])
    
    x_pos = range(len(distance_residuals))
    axs[1, 1].bar(x_pos, distance_residuals['mean'], color='purple', alpha=0.7)
    axs[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axs[1, 1].set_xticks(x_pos)
    axs[1, 1].set_xticklabels([f'{int(b.left)}-{int(b.right)}' 
                                for b in distance_residuals.index], rotation=45, ha='right')
    axs[1, 1].set_xlabel('Distance Range (miles)', fontsize=11)
    axs[1, 1].set_ylabel('Mean Residual (min)', fontsize=11)
    axs[1, 1].set_title('Prediction Bias by Flight Distance', fontsize=12)
    axs[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests on residuals
    from scipy.stats import normaltest
    
    # Normality test (on sample if too large)
    if len(residuals) > 5000:
        residual_sample = np.random.choice(residuals, 5000, replace=False)
    else:
        residual_sample = residuals
    
    _, normality_p = normaltest(residual_sample)
    
    result_text = f"""
- **Mean Residual**: {residuals.mean():.4f} min (should be ≈0 for unbiased model)
- **Residual Std Dev**: {residuals.std():.2f} min
- **Normality Test p-value**: {normality_p:.4f} ({'residuals approximately normal' if normality_p > 0.05 else 'residuals deviate from normality'})
- **Key Patterns**: 
  - {'Some systematic bias detected at specific hours; consider hour-specific features.' if hour_residuals['mean'].abs().max() > 2 else 'Minimal time-of-day bias.'}
  - {'Distance-dependent errors suggest non-linear distance effects.' if distance_residuals['mean'].abs().max() > 2 else 'Consistent performance across distances.'}
- **Recommendation**: {'Model is well-calibrated with random errors.' if abs(residuals.mean()) < 0.5 and normality_p > 0.01 else 'Consider additional features or model complexity to capture systematic patterns.'}
"""
    display_result("Residual Analysis Summary", result_text)
    
    return {
        'residuals': residuals,
        'mean_residual': residuals.mean(),
        'std_residual': residuals.std(),
        'normality_p': normality_p
    }



#--------------------------------
# Main Analysis Function
def run_all_analyses(df, selection=[1, 2, 3, 4], verbose=False):
    """
    Run all machine learning analyses on the flight data.
    

        Which analyses to run (1-4)
    verbose : bool
    """
    results = {}
    
    if 1 in selection:
        results['feature_importance'] = analyze_feature_importance(df)
        print("\n\n")
    
    if 2 in selection:
        results['model_comparison'] = compare_models_with_cv(df)
        print("\n\n")
    
    if 3 in selection:
        results['learning_curves'] = analyze_learning_curves(df)
        print("\n\n")
    
    if 4 in selection:
        results['residuals'] = analyze_residuals(df)
        print("\n\n")
    
    if verbose:
        return results
    else:
        return None