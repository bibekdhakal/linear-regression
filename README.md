# Parkinson's Disease Regression Analysis

This repository contains Python code for analyzing Parkinson's Disease data using regression techniques. The code performs the following tasks:

1. Data Loading and Preprocessing:
   - Import necessary libraries such as pandas, numpy, matplotlib, seaborn, and others for data manipulation, visualization, and modeling.
   - Load a CSV file named "po2_data.csv" into a Pandas DataFrame called 'df.'
   - Replace values in the 'sex' column, converting 'Male' to 0 and 'Female' to 1.
2. Linear Regression Modeling:
   - Extract feature variables (X) and target variable (y) from the DataFrame 'df.'
   - Split the dataset into training and testing sets using the train_test_split function.
   - Create a Linear Regression model using scikit-learn's LinearRegression class.
   - Fit the model to the training data.
   - Predict the target variable for the test data.
   - Calculate performance metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), normalized RMSE, and R-squared (R^2) for the linear regression model.
   - Also, create a baseline prediction using the mean value of the target variable.
3. Model Evaluation and Comparison:
   - Evaluate the performance of the linear regression model by calculating performance metrics.
   - Evaluate the performance of a dummy baseline model that predicts the mean value of the target variable.
   - Save the cleaned DataFrame to a CSV file.
4. Feature Engineering:
   - Select numeric columns and create scatter plots of these numeric features against the 'total_updrs' column and 'motor_updrs' column separately.
   - Save the scatter plots to specific directories.
5. Multicollinearity Analysis:
   - Calculate the Variance Inflation Factor (VIF) to check for multicollinearity among the feature variables.
   - Drop certain variables based on high VIF values to mitigate multicollinearity.
6. Principal Component Analysis (PCA):
   - Standardize the feature variables.
   - Perform Principal Component Analysis (PCA) to reduce the dimensionality of the data.
   - Save the transformed data to a CSV file.
7. Power Transformation:
   - Preprocess the feature variables by imputing missing values, standardizing, and applying power transformation (Yeo-Johnson method).
   - Save the preprocessed data to a CSV file.
8. Visualization of Correlation Matrix:
   - Calculate the correlation matrix for the DataFrame.
   - Create a heatmap using seaborn to visualize the correlation matrix.
   - Save the heatmap as an image.
9. Build and Evaluate Linear Regression using Statsmodels:
   - Fit an Ordinary Least Squares (OLS) regression model using the statsmodels library.
   - Predict the target variable using the regression model and obtain model statistics.
10. Rebuild and Reevaluate Linear Regression using Statsmodels with Collinearity Fixed:
    - Perform feature scaling to standardize the features.
    - Fit OLS regression models for both 'motor_updrs' and 'total_updrs' after transformation and scaling.
    - Evaluate the model performance and print model summaries.
11. Comparison of Linear Regression Models on Different Datasets:
    - Load different datasets from CSV files.
    - Create Linear Regression models for 'motor_updrs' and 'total_updrs' for each dataset.
    - Calculate and compare performance metrics (R-squared, MAE, MSE, RMSE) for the models.
    - Generate bar plots to visualize the comparison.

## Dependencies

Make sure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Usage

### Clone this repository:

git clone https://github.com/yourusername/parkinsons-regression.git
cd parkinsons-regression

#### Run the Python script:

python regression.py

1. The code will load and preprocess the Parkinson's Disease dataset, build and evaluate linear regression models, analyze multicollinearity, perform feature engineering, and visualize the correlation matrix.

2. Model comparison and performance metrics will be displayed and saved in the 'scatter_plots' and 'linear_regression_comparison.png' files.

## Files

1. regression.py: The main Python script containing the regression analysis code.
2. cleaned_data: Original Data before preprocessing.
3. cleaned_data.csv: Dataset after processing.
4. po2_after_PCA.csv: Dataset after applying Principal Component Analysis.
5. po2_after_VIF.csv: Dataset after removing variables with high Variance Inflation Factor.
6. po2_after_yeo-johnson.csv: Dataset after power transformation.
7. po2_after_rescaling.csv: Dataset after standardization and rescaling.
8. scatter_plots: Directory containing scatter plots of numeric data against 'motor_updrs' and 'total_updrs'.
9. linear_regression_comparison.png: Comparison plot of linear regression models on different datasets.

## Results

The results of the regression analysis, including model performance metrics and visualizations, can be found in the 'scatter_plots' directory and 'linear_regression_comparison.png'.
