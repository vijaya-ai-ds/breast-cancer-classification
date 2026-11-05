
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# Application title
st.title("Breast Cancer Classification")
st.write(
    "Evaluate five machine learning models "
    "using an uploaded test dataset."
)

# Model paths
model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}
# Upload dataset
uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

# Select model
selected_model = st.selectbox(
    "Select a classification model",
    list(model_paths.keys())
)

if uploaded_file is not None:

    # Read uploaded data
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(test_df.head())

    if "diagnosis" not in test_df.columns:
        st.error(
            "The uploaded CSV must contain a diagnosis column."
        )

    elif not os.path.exists(model_paths[selected_model]):
        st.error(
            "Model file not found. Save the trained models first."
        )

    else:
        # Separate features and target
        X_test = test_df.drop(columns=["diagnosis"])
        y_test = test_df["diagnosis"]

        # Load selected model
        model = joblib.load(model_paths[selected_model])

        # Check feature compatibility
        expected_features = list(model.feature_names_in_)
        actual_features = list(X_test.columns)

        if set(expected_features) != set(actual_features):
            st.error(
                "The uploaded dataset does not contain "
                "the required model features."
            )

        else:
            # Match the training feature order
            X_test = X_test[expected_features]

            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            precision = precision_score(
                y_test, y_pred, zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, zero_division=0
            )
            mcc = matthews_corrcoef(y_test, y_pred)

            # Display selected model
            st.subheader(f"Results: {selected_model}")

            # Display metrics
            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots()

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Benign", "Malignant"]
            )

            disp.plot(ax=ax, colorbar=False)
            st.pyplot(fig)
            plt.close(fig)

            # Download predictions
            output_df = test_df.copy()
            output_df["Predicted_Diagnosis"] = y_pred
            output_df["Malignant_Probability"] = y_prob

            st.download_button(
                "Download Predictions",
                data=output_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

else:
    st.info(
        "Upload test_data.csv to evaluate the selected model."
    )
