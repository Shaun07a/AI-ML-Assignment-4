# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective

The objective of this project is to develop a K-Nearest Neighbors (KNN) classification model to predict whether a breast tumor is malignant or benign based on diagnostic measurements. The project demonstrates data preprocessing, feature scaling, model training, prediction, and evaluation.

---

## Dataset Link

Breast Cancer Wisconsin Diagnostic Dataset

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

1. Load the dataset using Pandas.
2. Display the first five records, dataset information, and summary statistics.
3. Check for missing values.
4. Remove unnecessary columns (`id` and `Unnamed: 32`).
5. Encode the target variable (`diagnosis`).
6. Standardize the feature values using `StandardScaler`.
7. Split the dataset into 80% training and 20% testing sets.
8. Train a K-Nearest Neighbors classifier with **K = 5**.
9. Predict the class labels for the test dataset.
10. Evaluate the model using Accuracy Score, Precision, Recall, F1-Score, and a Confusion Matrix.

---

## Results

The KNN classifier achieved high classification performance on the Breast Cancer Wisconsin Diagnostic Dataset. The evaluation metrics indicate that the model accurately distinguishes between malignant and benign tumors, and the confusion matrix confirms that only a few samples are misclassified.

---

## Conclusion

This project demonstrates the effectiveness of K-Nearest Neighbors for breast cancer classification. Standardizing the feature values is essential because KNN is a distance-based algorithm. Feature scaling improves the model's ability to classify tumors accurately. Although KNN performs well on this dataset, one limitation is that prediction becomes slower as the size of the dataset increases because the algorithm compares each new sample with all training samples.
