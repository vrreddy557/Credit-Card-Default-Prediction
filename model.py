import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set style for EDA plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    """Fetching the classic UCI Credit Card Default Dataset from a raw public URL."""
    url = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Credit%20Default.csv"
    print(f"[*] Downloading dataset from: {url}")
    try:
        df = pd.read_csv(url)
        print(f"[+] Successfully loaded dataset. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[-] Error loading dataset: {e}")
        raise

def clean_and_preprocess(df):
    """Cleaning columns and structures features/labels according to the dataset schema."""
    print("[*] Preprocessing and profiling data...")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f" -> Missing values found: {missing}")

    target_col = None
    for col in df.columns:
        if 'default' in col.lower():
            target_col = col
            break
            
    if not target_col:
        # Fallback if specific naming varies
        target_col = df.columns[-1]
    
    print(f" -> Detected target column: '{target_col}'")
    
    # Isolating features and targets
    X = df.drop(columns=[target_col])
    # Removing high-cardinality index columns if present
    if 'id' in X.columns[0].lower():
        X = X.drop(columns=[X.columns[0]])
        
    y = df[target_col]
    
    # Converting categorical variables via One-Hot Encoding if string types exist
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y, target_col

def run_eda(df, target_col):
    """Performing Exploratory Data Analysis and saving distributions to disk."""
    print("[*] Running EDA and saving visualizations...")
    os.makedirs('plots', exist_ok=True)
    
    # 1. Target Distribution Plot
    plt.figure()
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title('Distribution of Credit Card Defaults (Class Imbalance View)')
    plt.xlabel('Default Status (0 = Kept Up, 1 = Defaulted)')
    plt.ylabel('Count')
    plt.savefig('plots/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation Heatmap for numerical features
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap Matrix')
    plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[+] EDA completed. Charts saved inside the 'plots/' directory.")

def train_evaluate_kfold(X, y):
    """Executing a 5-Fold Stratified Cross-Validation pipeline to counter leakage."""
    print("[*] Initializing 5-Fold Stratified Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    oof_probs = np.zeros(len(X))
    feature_importances = np.zeros(X.shape[1])
    
    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        print(f" -> Processing Fold {fold}/5...")
        
        # Splitting split partitions
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit scaler on training fold only to eliminate lookahead leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Instantiating Random Forest with class balancing
        model = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Accumulating predictions
        oof_preds[val_idx] = model.predict(X_val_scaled)
        oof_probs[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
        
        # Tracking feature significance
        feature_importances += model.feature_importances_ / skf.n_splits
        fold += 1

    print("\n================== OOF PERFORMANCE REPORT ==================")
    print(classification_report(y, oof_preds))
    
    auc_score = roc_auc_score(y, oof_probs)
    print(f"Overall Out-of-Fold ROC-AUC Score: {auc_score:.4f}")
    print("============================================================\n")
    
    # Saving Final Performance Evaluation Metrics Visualizations
    save_performance_plots(y, oof_preds, oof_probs, X, feature_importances)

def save_performance_plots(y_true, oof_preds, oof_probs, X, feature_importances):
    """Generating and saving the Confusion Matrix and ROC Curve."""
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, oof_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    plt.title('Out-of-Fold Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, oof_probs)
    auc_val = roc_auc_score(y_true, oof_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature Importance Plot (Top 10)
    feat_imp = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure()
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='mako')
    plt.title('Top 10 Most Predictive Features')
    plt.xlabel('Mean Decrease in Impurity (Gini Significance)')
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    raw_df = load_data()
    X_clean, y_clean, target_name = clean_and_preprocess(raw_df)
    run_eda(raw_df, target_name)
    train_evaluate_kfold(X_clean, y_clean)
    print("[+] Entire pipeline executed perfectly without issues.")
