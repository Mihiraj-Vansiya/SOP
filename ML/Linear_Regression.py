# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to load the dataset
def load_dataset(data_file):
    data = pd.read_csv(data_file)
    return data

# Function to perform train-test split
def perform_train_test_split(data, target_column):
    X, y = data.drop(columns=[target_column]), data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train the Linear Regression model
def train_linear_regression_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    return y_pred

# Function to visualize the model
def plot_model(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

if __name__ == "__main__":
    # Call the function with the data file
    data_file = 'Boston_Housing.csv'
    target_column = 'medv'  # Replace with the actual target column name

    # Load dataset
    data = load_dataset(data_file)

    # Perform train-test split
    X_train, X_test, y_train, y_test = perform_train_test_split(data, target_column)

    # Train the Linear Regression model and evaluate
    model = train_linear_regression_model(X_train, X_test, y_train, y_test)

    # Evaluate the Model
    y_pred = evaluate_model(model,X_test,y_test)

    # Visualize the model
    plot_model(y_test,y_pred)
