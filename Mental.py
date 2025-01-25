import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
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
        target_column = st.sidebar.selectbox("Select target column:", df.columns)

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(exclude=np.number).columns

        # Impute missing values separately for numerical and categorical features
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])


        # Create a column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        # Apply preprocessing
        X = preprocessor.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



        st.write("### Preprocessed Dataset (Example - first 5 rows):")
        st.write(pd.DataFrame(X[:5, :]).head()) #Show a sample of preprocessed data


        # --- Model Training and Evaluation ---
        st.sidebar.header("Model Selection")
        model_options = ["Decision Tree", "Random Forest", "Logistic Regression"]
        selected_models = st.sidebar.multiselect("Select models to train", model_options, default=model_options)

        results = {}
        models = {}

        for model_name in selected_models:
            if model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy
            models[model_name] = model

            # Add Classification Report
            report = classification_report(y_test, y_pred)
            st.write(f"### {model_name} Classification Report:")
            st.text(report)

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
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')#, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_) #Removed label encoding here as it's handled by OneHotEncoder
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for {best_model_name}')
            st.pyplot(fig)

        # Removed Histograms -  Not very useful for high-dimensional data and may be misleading


        # --- Prediction Section ---
        st.sidebar.header("Make Predictions")
        uploaded_test_file = st.sidebar.file_uploader("Upload test data (CSV format) for predictions", type=["csv"])

        if uploaded_test_file is not None:
            try:
                # Load test data
                test_df = pd.read_csv(uploaded_test_file)
                st.write("### Uploaded Test Data", test_df.head())

                # Preprocess test data (using the same preprocessor as training data)
                test_X = test_df.drop(target_column, axis=1)  #assuming same columns as training data
                test_X = preprocessor.transform(test_X)

                # Make predictions
                predictions = best_model.predict(test_X)
                st.write("### Predictions")
                st.write(pd.DataFrame({"Prediction": predictions}))

            except Exception as e:
                st.error(f"An error occurred while making predictions: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file.")

from sklearn.compose import ColumnTransformer
