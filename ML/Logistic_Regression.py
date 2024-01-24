import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    # Load the dataset
    X, y = load_dataset()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Standardize features
    X_train_std, X_test_std = standardize_data(X_train, X_test)

    # Train Logistic Regression model
    model = train_logistic_regression(X_train_std, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_std, y_test)

