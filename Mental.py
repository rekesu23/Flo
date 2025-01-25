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

st.title("Machine Learning Model Training App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df.head()) #Display first few rows

        # --- Data Preprocessing ---
        target_column = df.columns[-1]  # Assumes target is the last column

        df.fillna(method='bfill', inplace=True) #Backfill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        label_encoder = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = label_encoder.fit_transform(df[column])

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- Model Training and Evaluation ---
        st.subheader("Model Selection")
        model_choice = st.selectbox("Choose a model:", ("Decision Tree", "Random Forest", "Logistic Regression"))

        # Initialize models based on selection
        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000) #Increased max_iter for convergence

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")


            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

        except ValueError as e:
            st.error(f"Error during model training or prediction: {e}")
        except Exception as e:
            st.exception(e)


        # Histograms of Features
        st.subheader("Histograms of Features")
        for col in X.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col])
            plt.title(f'Histogram of {col}')
            st.pyplot(plt)


    except FileNotFoundError:
        st.error("Error: No file uploaded.")
    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("Error: Could not parse the uploaded file. Please ensure it's a valid CSV.")
    except KeyError:
        st.error("Error: The uploaded file does not contain the target column.")
    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")

else:
    st.info("Please upload a CSV file.")
