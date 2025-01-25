import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Machine Learning App")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset", df)

        # --- Data Preprocessing ---
        st.sidebar.header("Data Preprocessing")
        target_column = st.sidebar.selectbox("Select the target column", df.columns)

        # Handle missing values
        missing_strategy = st.sidebar.selectbox("Missing value handling", ["Backfill", "Forward Fill", "Drop Rows", "Mean Imputation"])
        if missing_strategy == "Backfill":
            df.fillna(method='bfill', inplace=True)
        elif missing_strategy == "Forward Fill":
            df.fillna(method='ffill', inplace=True)
        elif missing_strategy == "Drop Rows":
            df.dropna(inplace=True)
        elif missing_strategy == "Mean Imputation":
            df.fillna(df.mean(), inplace=True)

        # Convert object columns to numeric if possible
        label_encoder = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    df[col] = label_encoder.fit_transform(df[col])

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split data into training and testing sets
        test_size = st.sidebar.slider("Test set size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Scale the data
        scale_data = st.sidebar.checkbox("Scale Features", value=True)
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        st.write("### Preprocessed Dataset", df)

        # --- Model Training and Evaluation ---
        st.sidebar.header("Model Selection")
        model_options = ["Decision Tree", "Random Forest", "Logistic Regression"]
        selected_models = st.sidebar.multiselect("Select models to train", model_options, default=model_options)

        results = {}
        models = {}

        if "Decision Tree" in selected_models:
            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            dt_pred = dt.predict(X_test)
            dt_accuracy = accuracy_score(y_test, dt_pred)
            results["Decision Tree"] = dt_accuracy
            models["Decision Tree"] = dt

        if "Random Forest" in selected_models:
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            results["Random Forest"] = rf_accuracy
            models["Random Forest"] = rf

        if "Logistic Regression" in selected_models:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_accuracy = accuracy_score(y_test, lr_pred)
            results["Logistic Regression"] = lr_accuracy
            models["Logistic Regression"] = lr

        if results:
            st.write("### Model Accuracy")
            for model_name, accuracy in results.items():
                st.write(f"{model_name}: {accuracy:.2f}")

            best_model_name = max(results, key=results.get)
            best_model = models[best_model_name]
            st.write(f"### Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.2f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, best_model.predict(X_test))
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for {best_model_name}')
            st.pyplot(fig)

        # Visualizations
        st.write("### Feature Histograms")
        for col in X.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to get started.")
