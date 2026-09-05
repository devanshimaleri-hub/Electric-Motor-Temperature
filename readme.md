# Electric Motor Temperature Prediction

## Overview

This project develops a machine learning regression pipeline for predicting **stator winding temperature** in an electric motor.

The objective is to use measurable motor operating and thermal parameters to accurately estimate the stator winding temperature. Accurate temperature prediction can support motor monitoring, thermal management, and condition-aware operation.

The project follows a complete data-science workflow:

**Data Cleaning → Exploratory Data Analysis → Feature Engineering → Profile-Based Data Splitting → Model Training → Model Comparison → Hyperparameter Tuning → Final Testing → Error Analysis → Feature Importance → Model Saving**

---

## Project Structure

```text
Electric-Motor-Temperature/
│
├── data/
│   └── raw/
│       ├── archive (9).zip
│       └── measures_v2.csv
│
├── graphs/
│   ├── ambient_vs_stator_winding.png
│   ├── coolant_vs_stator_winding.png
│   ├── correlation_heatmap.png
│   ├── i_d_vs_stator_winding.png
│   ├── i_q_vs_stator_winding.png
│   ├── motor_speed_vs_stator_winding.png
│   ├── pm_vs_stator_winding.png
│   ├── profile_observation_counts.png
│   ├── stator_tooth_vs_stator_winding.png
│   ├── stator_yoke_vs_stator_winding.png
│   ├── torque_vs_stator_winding.png
│   ├── actual_vs_predicted.png
│   ├── residual_analysis.png
│   └── feature_importance.png
│
├── models/
│   └── final_temperature_model.pkl
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt

Dataset

The working dataset contains approximately 1.33 million observations across 69 motor profiles.

The target variable is:

stator_winding

The main input features used in the project include:

u_q
coolant
u_d
stator_tooth
motor_speed
i_d
i_q
pm
stator_yoke
ambient
torque

profile_id is used as a grouping variable for profile-based splitting rather than being treated as a normal predictive feature.

Dataset Quality

The data-cleaning stage verified:

69 unique motor profiles
0 missing values
0 duplicate rows

The number of observations varies between motor profiles, so the project also examined the observation distribution across profiles.

1. Data Cleaning

Notebook:

notebooks/01_data_cleaning.ipynb

The first notebook prepares the raw motor measurements for analysis.

The workflow includes:

Loading the raw CSV data
Inspecting the dataset shape and columns
Checking data types
Checking missing values
Checking duplicate observations
Examining descriptive statistics
Inspecting the motor profile structure
Preparing the dataset for downstream analysis

The final working dataset contains approximately 1.33 million rows and 69 motor profiles.

2. Exploratory Data Analysis

Notebook:

notebooks/02_eda.ipynb

The EDA stage investigates relationships between motor operating variables and stator winding temperature.

The project visualizes relationships such as:

Ambient temperature vs stator winding temperature
Coolant temperature vs stator winding temperature
Stator tooth temperature vs stator winding temperature
Stator yoke temperature vs stator winding temperature
Permanent-magnet temperature (pm) vs stator winding temperature
Motor speed vs stator winding temperature
Torque vs stator winding temperature
i_d vs stator winding temperature
i_q vs stator winding temperature

A correlation matrix was also generated to examine relationships among the motor parameters.

Important EDA Observations

The correlation analysis shows particularly strong relationships between stator winding temperature and several thermal variables, including:

stator_tooth
stator_yoke
pm
coolant

The analysis also shows relationships between several motor operating variables, indicating that stator winding temperature depends on multiple interacting parameters.

3. Feature Engineering and Data Splitting

Notebook:

notebooks/03_feature_engineering.ipynb

A key design decision in this project is the use of profile-based data splitting.

Instead of randomly splitting individual rows, complete motor profiles are assigned to the training, validation, and test sets.

This is important because observations belonging to the same motor profile can be highly correlated. A random row split could allow very similar observations from the same profile to appear in both training and testing data, resulting in overly optimistic performance estimates.

Final Profile Split

The final dataset was divided into:

Training profiles: 55
Validation profiles: 7
Test profiles: 7

The split was checked to ensure there was no overlap between the three profile groups.

Therefore, the final test set represents motor profiles that were not used during model training.

4. Model Training

Notebook:

notebooks/04_model_training.ipynb

The task is formulated as a supervised regression problem.

Input:
Motor operating and thermal measurements

Output:
Stator winding temperature
Baseline Model

A DummyRegressor using the mean prediction strategy was used as a baseline.

This provides a simple reference point for evaluating whether the machine learning models provide meaningful predictive improvement.

Models Evaluated
Linear Regression

A basic linear regression model used to establish how well the target can be explained through linear relationships between the input variables and stator winding temperature.

Ridge Regression

A regularized linear regression model using L2 regularization. It helps improve stability when input variables are correlated.

Random Forest Regressor

An ensemble of decision trees capable of learning nonlinear relationships and interactions between motor parameters.

HistGradientBoosting Regressor

A gradient-boosting model that builds trees sequentially, with each stage attempting to improve upon previous prediction errors.

HistGradientBoosting was selected for further tuning as the nonlinear boosting approach in the final pipeline.

5. Evaluation Metrics

Because this project predicts a continuous temperature value, it is a regression problem.

Therefore, classification metrics such as accuracy, precision, recall, and F1-score are not appropriate.

The main evaluation metrics are:

MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted temperatures.

Lower values are better.

MAE = average(|actual - predicted|)

Since temperature is measured in °C, MAE is also expressed in °C and is easy to interpret.

RMSE — Root Mean Squared Error

RMSE measures the square root of the average squared prediction error.

It gives greater importance to larger errors than MAE.

Lower values are better.

RMSE = sqrt(mean((actual - predicted)^2))
R² — Coefficient of Determination

R² measures how much of the variation in the target temperature is explained by the model.

Higher values are better.

An R² value of 1 represents a perfect prediction.

6. Model Comparison

The initial validation comparison produced the following results:

Model	MAE (°C)	RMSE (°C)	R²
Ridge Regression	1.322641	1.772158	0.996595
Linear Regression	1.322641	1.772158	0.996595
HistGradientBoosting	1.268823	1.816728	0.996421
Random Forest	1.437451	2.110603	0.995170
Baseline	25.905484	30.605688	-0.015676

The baseline performs significantly worse than the trained models, demonstrating that the motor features contain substantial predictive information.

All trained models achieve very high R² values. HistGradientBoosting was selected for further tuning because it provides a strong nonlinear modeling approach.

7. Hyperparameter Tuning

The selected HistGradientBoosting model was tuned to improve its predictive performance.

The tuned model was evaluated against the original HistGradientBoosting model using MAE, RMSE, and R².

The tuned model achieved approximately:

MAE: 1.2769 °C
RMSE: 1.8000 °C
R²: 0.996487

The tuned model was then used as the final model for evaluation on the completely unseen test profiles.

8. Final Test Evaluation

The final model was evaluated on the test set only after model selection and tuning.

Final Test Performance
Metric	Result
MAE	1.145 °C
RMSE	1.644 °C
R²	0.99696

These results indicate that the final model predicts stator winding temperature with a very small average error.

An R² of approximately 0.997 indicates that the model explains nearly all of the observed variation in stator winding temperature in the unseen test profiles.

9. Actual vs Predicted Analysis

The project includes an actual-vs-predicted visualization:

graphs/actual_vs_predicted.png

This graph compares the measured stator winding temperatures with the temperatures predicted by the final model.

A strong model should produce points concentrated around the diagonal line representing:

Actual Temperature = Predicted Temperature

This visualization provides a graphical confirmation of the numerical evaluation metrics.

10. Residual Analysis

Residual analysis is used to understand the model's prediction errors.

The residual is defined as:

Residual = Actual Temperature - Predicted Temperature

The project generates a residual plot:

graphs/residual_analysis.png

Ideally, residuals should be distributed randomly around zero without strong systematic patterns.

Residual analysis helps identify potential systematic errors that may not be obvious from MAE, RMSE, or R² alone.

11. Feature Importance

Permutation importance is used to estimate how strongly each input feature contributes to the predictive performance of the final model.

The feature importance visualization is stored as:

graphs/feature_importance.png

Features with larger permutation importance values have a greater influence on the model's predictions.

This provides an additional layer of interpretability and helps identify which motor parameters are most informative for estimating stator winding temperature.

12. Final Model

The trained final model is saved as:

models/final_temperature_model.pkl

The model can be loaded later using joblib without retraining.

Example:

import joblib

model = joblib.load("models/final_temperature_model.pkl")

Predictions can then be generated using the same feature structure used during training.

13. Reproducibility

Create a Python virtual environment:

Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate

Install the project dependencies:

pip install -r requirements.txt

Launch Jupyter:

jupyter notebook

Alternatively, open the project in VS Code and select the project's .venv Python environment as the Jupyter kernel.

Run the notebooks in this order:

01_data_cleaning.ipynb
        ↓
02_eda.ipynb
        ↓
03_feature_engineering.ipynb
        ↓
04_model_training.ipynb
14. Technologies Used
Python 3.11
Pandas — data manipulation
NumPy — numerical computation
Matplotlib — visualization
Seaborn — statistical visualization
Scikit-learn — machine learning, evaluation and tuning
Joblib — model persistence
Jupyter Notebook — experimentation and analysis
Git / GitHub — version control
15. Key Project Decisions
Why profile-based splitting?

Measurements belonging to the same motor profile can be highly correlated.

Splitting by profile prevents related observations from the same profile from appearing in both training and testing sets.

This gives a more realistic estimate of how the model generalizes to unseen motor profiles.

Why regression metrics instead of accuracy, recall, or F1-score?

The target variable is a continuous temperature value.

For example:

87.42 °C
91.17 °C
103.54 °C

These are not class labels.

Therefore, regression metrics such as MAE, RMSE, and R² are appropriate, while accuracy, recall, and F1-score are classification metrics and are not applicable to the main prediction task.

Why use a baseline?

The baseline provides a simple reference point.

If the machine learning models cannot significantly outperform the baseline, then the input features may not contain useful predictive information.

In this project, the machine learning models dramatically outperform the baseline.

Why compare multiple models?

Different algorithms capture different types of relationships.

The project therefore compares:

Linear models
Regularized linear models
Tree ensembles
Gradient boosting

This provides a stronger basis for selecting the final model.

Why tune the model?

Hyperparameter tuning can improve the selected model's ability to generalize to unseen motor profiles.

16. Conclusion

This project successfully developed an end-to-end machine learning pipeline for predicting stator winding temperature in an electric motor.

The workflow includes:

Data cleaning
Exploratory data analysis
Correlation analysis
Feature preparation
Profile-based train/validation/test splitting
Leakage prevention
Baseline modeling
Multiple regression models
Model comparison
Hyperparameter tuning
Final unseen-test evaluation
Actual-vs-predicted analysis
Residual analysis
Feature importance
Model persistence

The final model achieved:

MAE  = 1.145 °C
RMSE = 1.644 °C
R²   = 0.99696

These results demonstrate that the available motor operating and thermal measurements provide strong predictive information about stator winding temperature.

The use of profile-based splitting also provides a more realistic evaluation by testing the final model on motor profiles that were not used during training.

The final trained model is available at:

models/final_temperature_model.pkl
Future Improvements

Possible future extensions include:

Developing a real-time temperature prediction interface
Creating a web dashboard for motor monitoring
Building an API for model inference
Evaluating prediction latency for real-time applications
Investigating time-series-aware validation
Adding prediction uncertainty estimation
Deploying the trained model as a lightweight prediction service
Exploring additional advanced boosting algorithms if required by the application
Author

Electric Motor Temperature Prediction

An end-to-end machine learning project covering data preparation, exploratory analysis, leakage-aware data splitting, regression modeling, hyperparameter tuning, evaluation, interpretation, and model persistence.