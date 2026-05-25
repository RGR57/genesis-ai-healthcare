import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):

    # =========================
    # Load Dataset
    # =========================
    df = pd.read_csv(file_path)

    # =========================
    # Shuffle Dataset
    # =========================
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # =========================
    # Clean String Columns
    # =========================
    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = df[column].astype(str)

            df[column] = df[column].str.strip()

    # =========================
    # Encode ALL Categorical Data
    # =========================
    df = pd.get_dummies(df)

    # =========================
    # Separate Features/Target
    # =========================
    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    feature_names = list(X.columns)

    # =========================
    # Convert to Numeric
    # =========================
    X = X.astype(float)

    # =========================
    # Feature Scaling
    # =========================
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # =========================
    # Train/Test Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled,
        y,

        test_size=0.30,

        stratify=y,

        random_state=42
    )

    return (
        (X_train, X_test, y_train, y_test),
        feature_names
    )