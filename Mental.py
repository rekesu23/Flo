import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Machine Learning Model Training App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df.head())

        # --- Data Preprocessing ---
        st.subheader("Data Preprocessing")
        target_column = st.selectbox("Select Target Column:", df.columns) #Let user choose target

        #Handle Missing Values (Improved)
        missing_values = df.isnull().sum()
        st.write("Missing Values per Column:")
        st.write(missing_values)
        if missing_values.sum() > 0:
          strategy = st.selectbox("Choose missing value strategy:", ["Drop rows with missing values", "Fill with mean/median/mode", "Fill with a constant"])
          if strategy == "Drop rows with missing values":
            df.dropna(inplace=True)
          elif strategy == "Fill with mean/median/mode":
            for col in df.columns:
              if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
              elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
          elif strategy == "Fill with a constant":
            constant_value = st.text_input("Enter constant value to fill missing data:")
            if constant_value:
              df.fillna(constant_value, inplace=True)

        #Feature Encoding (Improved - handles both numerical and categorical)
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=np.number).columns

        if len(categorical_cols) > 0:
          encoding_method = st.selectbox("Choose encoding method for categorical features:", ["One-Hot Encoding", "Label Encoding"])
          if encoding_method == "One-Hot Encoding":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
            df = df.drop(columns=categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
          elif encoding_method == "Label Encoding":
            label_encoder = LabelEncoder()
            for col in categorical_cols:
              df[col] = label_encoder.fit_transform(df[col])

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- Model Training and Evaluation ---
        st.subheader("Model Selection and Training")
        model_choice = st.selectbox("Choose a model:", ("Decision Tree", "Random Forest", "Logistic Regression"))

        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))


            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df[target_column].unique(), yticklabels=df[target_column].unique()) #Use unique labels
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

        except ValueError as e:
            st.error(f"Error during model training or prediction: {e}")
        except Exception as e:
            st.exception(e)

    except FileNotFoundError:
        st.error("Error: No file uploaded.")
    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("Error: Could not parse the uploaded file. Please ensure it's a valid CSV.")
    except KeyError:
        st.error("Error: Target column not found in the dataset.")
    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")

else:
    st.info("Please upload a CSV file.")
