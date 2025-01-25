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

        # ... (rest of your model training and evaluation code remains the same) ...

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file.")
