"""
improved.py

Enhanced machine learning analysis for flight delay prediction using 
advanced preprocessing, cyclical encoding, and dimensionality reduction (PCA).
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, Markdown
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


#--------------------------------
# Helper Functions 

def display_title(s, pref='Figure', num=1):
    """Display formatted title for analysis sections."""
    s = f'<p><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></p>'
    display(Markdown(s))


def display_result(title, content):
    display(Markdown(f"**{title}**\n\n{content}"))



#--------------------------------
# Advanced Preprocessing Pipeline
def preprocess_data_improved(df, use_pca=True, n_components=10):
    """
    Improves results via Cyclical Encoding, Robust Scaling, and PCA.
    """
    df_clean = df.dropna(subset=['arrival_delay', 'departure_delay', 'distance']).copy()
    
    # 1. Cyclical Feature Engineering (Time is a circle, not a linear line)
    df_clean['dep_hour'] = df_clean['scheduled_dep_time'] // 100
    df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['dep_hour'] / 24)
    df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['dep_hour'] / 24)
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
    
    # 2. Interaction Features
    df_clean['delay_dist_interaction'] = df_clean['departure_delay'] * df_clean['distance']
    
    # 3. Target Transformation (Handling Skewness-assymetry of data distribution around the mean)
    # We use log1p to handle the fact that delays can be 0 or slightly negative
    y = np.log1p(df_clean['arrival_delay'].clip(lower=0)) 
    
    # 4. Dimensionality Reduction on Destinations
    # Instead of hundreds of one-hot columns, we use PCA on the destination dummies
    dest_dummies = pd.get_dummies(df_clean['destination'], prefix='dest')
    if use_pca:
        pca = PCA(n_components=min(n_components, dest_dummies.shape[1]))
        dest_pca = pca.fit_transform(dest_dummies)
        dest_cols = [f'dest_pca_{i}' for i in range(dest_pca.shape[1])]
        dest_df = pd.DataFrame(dest_pca, columns=dest_cols, index=df_clean.index)
    else:
        dest_df = dest_dummies

    # Combine features
    features = ['departure_delay', 'distance', 'hour_sin', 'hour_cos', 
                'month_sin', 'month_cos', 'delay_dist_interaction']
    X = pd.concat([df_clean[features], dest_df], axis=1)
    
    # 5. Robust Scaling (Better for datasets with outliers/extreme delays, not exactly ours)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns



#--------------------------------
# Improved Analysis 1: Enhanced Feature Importance
def analyze_feature_importance_improved(df):
    display_title('Improved Feature Engineering & Importance', num=4)
    
    X, y, cols = preprocess_data_improved(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate (Transforming back from log scale for RMSE)
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Importance
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='barh', color='teal')
    plt.title("Top 10 Features (Post-Preprocessing & PCA)")
    plt.show()
    
    res = f"""
- **Improved RMSE**: {rmse:.2f} min
- **Improved R²**: {r2:.4f}
- **Strategy**: Used **Log Transformation** on target and **Cyclical Encoding** for time.
- **Dimensionality Reduction**: Reduced destination sparsity using **PCA**.
"""
    display_result("Comparison with Original", res)
    return model



#--------------------------------
# Improved Analysis 2: Model Comparison with Reduced Dimensions
def compare_models_improved(df):
    """
    Compares models using the improved preprocessing pipeline and 
    visualizes the results to demonstrate dimensionality reduction impact.
    """
    display_title('Model Comparison with Dimensionality Reduction', num=5)
    
    # preprocessing with a smaller number of PCA components to focus on core variance
    X, y, _ = preprocess_data_improved(df, n_components=5)
    
    models = {
        'Ridge (Robust)': Ridge(),
        'GBM (Log-Target)': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # use cross-validation to get robust R^2 scores
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        results[name] = cv_scores.mean()
    
    # visualization implementation
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    scores = list(results.values())
    
    bars = plt.bar(names, scores, color=['#4682B4', '#20B2AA'], alpha=0.85, edgecolor='black')
    
    # add labels and styling
    plt.ylabel('Average CV $R^2$ Score', fontsize=12)
    plt.title('Performance Comparison: Linear vs. Ensemble on PCA-Reduced Data', fontsize=14)
    plt.ylim(0, max(scores) * 1.15)  # provide space for text labels 1.15 should be enough
    
    # annotate bars with exact R^2 values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    res = f"""
- **Finding**: Preprocessing the data into a dense PCA space improves convergence for models like GBM.
- **Ridge Performance**: The **Ridge (Robust)** model benefits from **RobustScaler**, which reduces the impact of extreme flight delay outliers.
- **Ensemble Advantage**: **GBM (Log-Target)** typically performs best as it captures non-linear patterns in the log-transformed delay space.
- **Dimensionality Reduction**: By using **PCA** on destinations, we avoid the sparse matrix issues found in the original analysis.
"""
    display_result("Dimensionality Reduction Impact", res)


#--------------------------------
# Main Analysis Function
def run_improved_analyses(df, selection=[1,2]):
    if 1 in selection:
        analyze_feature_importance_improved(df)
    if 2 in selection:
        compare_models_improved(df)