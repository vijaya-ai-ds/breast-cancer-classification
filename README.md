# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

Breast cancer diagnosis is an important classification problem in healthcare. The objective of this project is to develop machine learning classification models that predict whether a breast tumour is Benign (B) or Malignant (M) using measurements of cell nuclei.

Five classification algorithms are implemented and evaluated using six performance metrics. A Streamlit web application is also developed to allow users to upload test data, select a model, and examine its predictions and evaluation results.

## 2. Dataset Description

This project uses the Breast Cancer Wisconsin dataset obtained from Kaggle.

- **Number of instances:** 569
- **Target column:** `diagnosis`
- **Target classes:** Benign (B) and Malignant (M)
- **Predictive features:** 30 numerical features
- **Identifier column:** `id`, which is not used as a predictive feature.

The features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, and concave points.

The dataset is suitable for binary classification because the target variable contains two classes. It satisfies the assignment's minimum requirements of 500 instances and 12 features.

## 3. GitHub Repository Link

**Repository:** [Breast Cancer Classification](https://github.com/vijaya-ai-ds/breast-cancer-classification)

## 4. Live Streamlit Application

**Streamlit App:** [Breast Cancer Classification – Live Application](https://2026ak05244.streamlit.app/)

The application allows users to upload a test CSV file, select one of the five trained models, and view the classification results.

## 5. Machine Learning Models Used

The following five classification algorithms are implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 6. Evaluation Metrics

The models are evaluated using the following six metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures how well the model distinguishes between the two classes.
- **Precision:** Measures the proportion of positive predictions that are correct.
- **Recall:** Measures the proportion of actual positive instances correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the agreement between actual and predicted classifications.

### 6.1 Model Performance Comparison

The following table presents the evaluation results obtained from the current test split.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### 6.2 Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 96.49% and an AUC score of 0.9960. It performed well in distinguishing between benign and malignant tumours. Its F1 score of 0.9512 indicates a good balance between precision and recall.

**2. Decision Tree Classifier**

The Decision Tree achieved an accuracy of 92.98%. Its accuracy and F1 score were lower than those of the other models in this comparison. However, it still provided good classification results.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 95.61% and a precision of 0.9744. Its recall was 0.9048, indicating that some actual positive cases were not identified. Overall, it showed good classification performance.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 93.86% and an AUC score of 0.9934. It achieved a precision of 1.0000 but a recall of 0.8333. This means that its positive predictions were correct on this test split, but it failed to identify some actual positive cases.

**5. Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 97.37%, a precision of 1.0000, an F1 score of 0.9630, and an MCC of 0.9442. Its AUC score was 0.9947. It provided the strongest overall performance by accuracy on the current test split.

### 6.3 Overall Winner

**Support Vector Machine (SVM) is the overall winner based on the current test results.**

It achieved the highest accuracy, F1 score, and MCC among the five evaluated models. It also achieved a precision of 1.0000 and a recall of 0.9286.

Logistic Regression achieved the highest AUC score of 0.9960. Therefore, SVM performed best by accuracy, while Logistic Regression achieved the highest AUC on this test split.

These results represent the current test split and do not guarantee the same performance on other datasets.

## 7. Project Structure

```text
breast-cancer-classification/
│
├── model/
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── app.py
├── data.csv
├── test_data.csv
├── ML_Assignment_2.ipynb
├── README.md
└── requirements.txt
```

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 9. Installation

Install the required dependencies using the following command:

```bash
python -m pip install -r requirements.txt
```

## 10. Running the Application

Open a terminal in the project directory and execute:

```bash
python -m streamlit run app.py
```

The application will open in a web browser.

## 11. Streamlit Application Features

The application provides the following features:

- Upload a test dataset in CSV format.
- Preview the uploaded dataset.
- Select one of the five trained classification models.
- Load the selected saved model.
- Generate predictions on the uploaded test data.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display a confusion matrix for evaluating classification performance.
- Download the prediction results as a CSV file.

The application uses saved models to make predictions without retraining them on the uploaded test dataset.

## 12. Conclusion

Five machine learning classification algorithms were implemented and evaluated on the Breast Cancer Wisconsin dataset using six performance metrics.

Among the evaluated models, Support Vector Machine achieved the highest accuracy of 97.37% and the highest F1 score of 0.9630. Logistic Regression achieved the highest AUC score of 0.9960.

The Streamlit application provides an interactive interface for uploading test data, selecting classification models, and examining their evaluation results.

This project demonstrates the use of machine learning for binary classification and the deployment of trained models through a web application.
