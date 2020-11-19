# DEMAND
An interactive dashboard based on a Linear Regression model for predicting demand for bike rentals in Washington, D.C.

An approach to Machine Learning with Linear Regression models. Through intensive Explorative Data Analysis, correlation analysis, and feature engineering I was able to construct robust models to estimate number of bike rentals at any particular time. GridSearchCV was used to attain the best fitting model and parameters.

<img src="https://github.com/brauliotegui/DEMAND/blob/main/dashdemo.gif">

## Models:
A few models were used and tested on this project to see which one would obtain the best results for this specific problem. Since I wanted to focus on Linear models only, the selected ones were Linear Regression, Ridge Regression, and Lasso Regression. GridSearchCV was also used in this project in order to achieve the best parameters for the models. The one performing best for this project was Ridge with the following pipeline attributes:
- PolynomialFeatures (degree=8)
- Ridge (alpha=0.0001, max_iter=3000, normalize=True)
- Metrics:
  - R2: 81.53 %
  - RMSLE: 0.60 %

## Tech used:
 - Python
 - Scikit-learn
 - GridSearchCV
 - Dash
 - Plotly
 - HTML
 - CSS
