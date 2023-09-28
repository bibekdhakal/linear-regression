# Parkinson's Disease Regression Analysis

This repository contains Python code for analyzing Parkinson's Disease data using regression techniques. The code performs the following tasks:

1. Data Loading and Preprocessing
2. Linear Regression Modeling
3. Model Evaluation and Comparison
4. Feature Engineering
5. Multicollinearity Analysis
6. Principal Component Analysis (PCA)
7. Power Transformation
8. Visualization of Correlation Matrix

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
