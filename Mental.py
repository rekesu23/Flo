import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Mental Disorder Prediction App")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset", df.head())

        # --- Data Preprocessing ---
        st.sidebar.header("Data Preprocessing")

        # Get target column (with error handling)
        try:
            target_column = st.sidebar.selectbox("Select target column ('SeriousDisorder' if present):", df.columns)
        except Exception as e:
            st.error(f"Error selecting target column: {e}. Please ensure your CSV has a suitable target column.")
            st.stop() # Stop execution if target selection fails


        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(exclude=np.number).columns

        # Create preprocessing pipeline (improved)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_cols),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
            ])

        #Improved Model Selection and Training
        st.sidebar.header("Model Selection")
        model_options = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        selected_model_name = st.sidebar.selectbox("Select a model:", list(model_options.keys()))
        model = model_options[selected_model_name]


        # Create and train the pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipeline.fit(X_train, y_train)

        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"### {selected_model_name} Results:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(report)

        #Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {selected_model_name}')
        st.pyplot(fig)


        # --- Prediction Section ---
        st.sidebar.header("Make Predictions")
        uploaded_test_file = st.sidebar.file_uploader("Upload test data (CSV format) for predictions", type=["csv"])

        if uploaded_test_file is not None:
            try:
                test_df = pd.read_csv(uploaded_test_file)
                st.write("### Uploaded Test Data", test_df.head())
                test_X = test_df.drop(target_column, axis=1, errors='ignore') #ignore errors if column not found

                predictions = pipeline.predict(test_X)
                st.write("### Predictions")
                st.write(pd.DataFrame({"Prediction": predictions}))

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file.")
