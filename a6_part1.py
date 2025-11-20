"""
Assignment 6 Part 1: Student Performance Prediction
Name: _______________
Date: _______________

This assignment predicts student test scores based on hours studied.
Complete all the functions below following the in-class ice cream example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the student scores data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # TODO: Load the CSV file using pandas
    data = pd.read_csv(filename)
    # TODO: Print the first 5 rows
    print("Student scores")
    # TODO: Print the shape of the dataset (number of rows and columns)
    
    # TODO: Print basic statistics (mean, min, max, etc.)
    
    # TODO: Return the dataframe
    pass


def create_scatter_plot(data):
    """
    Create a scatter plot to visualize the relationship between hours studied and scores
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    """
    # TODO: Create a figure with size (10, 6)
    plt.figure(figsize=[10, 6])
    # TODO: Create a scatter plot with Hours on x-axis and Scores on y-axis
    #       Use color='purple' and alpha=0.6
    plt.scatter(data['Hours'], data['score'])
    # TODO: Add x-axis label: 'Hours Studied'
    
    # TODO: Add y-axis label: 'Test Score'
    
    # TODO: Add title: 'Student Test Scores vs Hours Studied'
    
    # TODO: Add a grid with alpha=0.3
    
    # TODO: Save the figure as 'scatter_plot.png' with dpi=300
    
    # TODO: Show the plot
    pass


def split_data(data):
    """
    Split data into features (X) and target (y), then into training and testing sets
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Create X with the 'Hours' column (use double brackets to keep as DataFrame)
    
    # TODO: Create y with the 'Scores' column
    
    # TODO: Split the data using train_test_split with test_size=0.2 and random_state=42
    
    # TODO: Print how many samples are in training and testing sets
    
    # TODO: Return X_train, X_test, y_train, y_test
    pass


def train_model(X_train, y_train):
    """
    Create and train a linear regression model
    
    Args:
        X_train: training features
        y_train: training target values
    
    Returns:
        trained LinearRegression model
    """
    # TODO: Create a LinearRegression model
    model = LinearRegression()
    # TODO: Train the model using .fit()
    model.fit(X_train,y_train)
    # TODO: Print the coefficient (slope) and intercept
    
    # TODO: Return the trained model
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on test data
    
    Args:
        model: trained LinearRegression model
        X_test: testing features
        y_test: testing target values
    
    Returns:
        predictions array
    """
    # TODO: Make predictions using the model
    predictions = model.predict(X_test)
    # TODO: Calculate R² score using r2_score()
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    # TODO: Calculate Mean Squared Error using mean_squared_error()
    
    # TODO: Calculate Root Mean Squared Error (square root of MSE)
    
    # TODO: Print all three metrics with clear labels
    
    # TODO: Return the predictions
    pass


def visualize_results(X_train, y_train, X_test, y_test, predictions, model):
    """
    Visualize the model's predictions against actual values
    
    Args:
        X_train: training features
        y_train: training target values
        X_test: testing features
        y_test: testing target values
        predictions: model predictions on test set
        model: trained model (to plot line of best fit)
    """
    # TODO: Create a figure with size (12, 6)
    
    # TODO: Plot training data as blue scatter points with label 'Training Data'
    
    # TODO: Plot test data (actual) as green scatter points with label 'Test Data (Actual)'
    
    # TODO: Plot predictions as red X markers with label 'Predictions'
    
    # TODO: Create and plot the line of best fit
    #       Hint: Create a range of X values, predict Y values, then plot as a black line
    
    # TODO: Add x-axis label, y-axis label, and title
    
    # TODO: Add legend
    
    # TODO: Add grid with alpha=0.3
    
    # TODO: Save the figure as 'predictions_plot.png' with dpi=300
    
    # TODO: Show the plot
    pass


def make_prediction(model, hours):
    """
    Make a prediction for a specific number of hours studied
    
    Args:
        model: trained LinearRegression model
        hours: number of hours to predict score for
    
    Returns:
        predicted test score
    """
    # TODO: Reshape hours into the format the model expects: np.array([[hours]])
    
    # TODO: Make a prediction
    
    # TODO: Print the prediction with a clear message
    
    # TODO: Return the predicted score
    pass


if __name__ == "__main__":
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore the data
    data = load_and_explore_data('student_scores.csv')
    # Step 2: Visualize the relationship
    # TODO: Call create_scatter_plot() with the data
    create_scatter_plot(data)
    # Step 3: Split the data
    # TODO: Call split_data() and store the returned values
    
    # Step 4: Train the model
    # TODO: Call train_model() with training data
    
    # Step 5: Evaluate the model
    # TODO: Call evaluate_model() with the model and test data
    
    # Step 6: Visualize results
    # TODO: Call visualize_results() with all the necessary arguments
    
    # Step 7: Make a new prediction
    # TODO: Call make_prediction() for a student who studied 7 hours
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part1_writeup.md!")
    print("=" * 70) 
