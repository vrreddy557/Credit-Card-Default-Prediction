import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dataset Source: UCI Machine Learning Repository
DATA_URL = "https://raw.githubusercontent.com/amirshnll/default-of-credit-card-clients-classification/master/dataset.csv"

def run_pipeline():
    print("Fetching data from UCI source...")
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Preprocessing
    # Target: 'default.payment.next.month'
    target = 'default.payment.next.month'
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Random Forest on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_pipeline()