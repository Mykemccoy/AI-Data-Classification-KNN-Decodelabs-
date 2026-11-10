import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    # 1. Load and Understand Dataset
    # Ensure 'Dataset for Data Analytics.xlsx' is in the same folder as this script, 
    # or provide the full file path.
    file_path = 'Dataset for Data Analytics.xlsx'
    df = pd.read_excel(file_path)
    
    print("=== Dataset Summary ===")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print(df.head())

    # 2. Feature Engineering & Data Preparation
    # Define target label: 1 if TotalPrice is above median, 0 otherwise
    median_price = df['TotalPrice'].median()
    df['HighValueOrder'] = (df['TotalPrice'] > median_price).astype(int)

    features = ['Quantity', 'UnitPrice', 'ItemsInCart']
    X = df[features]
    y = df['HighValueOrder']

    # 3. Split into Training and Testing Sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling (Essential for KNN distance calculations)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (K-Nearest Neighbors)
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # 6. Evaluation
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Algorithm: K-Nearest Neighbors (k={k})")
    print(f"Accuracy Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()
