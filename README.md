#### &#x20;House Price Prediction Using Linear Regression, Ridge, and Lasso



##### &#x20;Project Overview



This project predicts house prices using machine learning regression models. The dataset contains housing information such as living      area, number of bedrooms, bathrooms, house grade, location, and other property features.



The project compares the performance of:



\* Linear Regression

\* Ridge Regression

\* Lasso Regression



using evaluation metrics such as RMSE and R² Score.



\---



##### Dataset



Dataset: House Prices Dataset



Total Records: 21,613



Target Variable: price



Features Used:



\* bedrooms

\* bathrooms

\* sqft\_living

\* sqft\_lot

\* floors

\* waterfront

\* view

\* condition

\* grade

\* sqft\_above

\* sqft\_basement

\* yr\_built

\* yr\_renovated

\* zipcode

\* lat

\* long

\* sqft\_living15

\* sqft\_lot15







##### &#x20;Project Workflow



&#x20;1. Data Loading



\* Loaded dataset using Pandas.



&#x20;2. Exploratory Data Analysis (EDA)



\* Dataset overview using head(), info(), and describe().

\* Correlation analysis.

\* Heatmap visualization.

\* Distribution and relationship plots.



&#x20;3. Data Preprocessing



\* Removed unnecessary columns.

\* One-Hot Encoding for categorical features.

\* Feature Engineering.

\* Train-Test Split.

\* Feature Scaling using StandardScaler.



&#x20;4. Model Building



Implemented:



\* Linear Regression

\* Ridge Regression

\* Lasso Regression



&#x20;5. Model Evaluation



Models were evaluated using:



\* Mean Absolute Error (MAE)

\* Root Mean Squared Error (RMSE)

\* R² Score



\---



##### &#x20;Results



| Model             | MAE       | RMSE      | R² Score |

| ----------------- | --------- | --------- | -------- |

| Linear Regression | 125209.78 | 203426.25 | 0.6956   |

| Ridge Regression  | 125207.87 | 203426.50 | 0.6956   |

| Lasso Regression  | 125209.33 | 203426.35 | 0.6956   |



\---



##### &#x20;Key Findings



\* Living area (sqft\_living) showed the strongest relationship with house price.

\* House grade and bathrooms were important predictors.

\* Ridge and Lasso performed similarly to Linear Regression.

\* The final model explained approximately 69.5% of the variation in house prices.



\---



##### &#x20;Technologies Used



\* Python

\* Pandas

\* NumPy

\* Matplotlib

\* Seaborn

\* Scikit-Learn

\* Jupyter Notebook



\---



##### &#x20;Future Improvements



\* Hyperparameter tuning

\* Cross-validation

\* Advanced feature engineering

\* Random Forest Regressor

\* XGBoost Regressor



\---



##### &#x20;Author



&#x20; Deepak S





