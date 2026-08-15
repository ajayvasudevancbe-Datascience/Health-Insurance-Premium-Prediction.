# Health-Insurance-Premium-Prediction.

This analysis applies **machine learning to health insurance premium prediction**. The objective is to estimate insurance charges based on customer characteristics such as age, BMI, number of children, sex, smoking status and region. The study also examines how adding relevant features affects the predictive performance of the models.

The methodology includes **data exploration, preprocessing, feature transformation and Linear Regression modeling**. Categorical variables such as sex, smoker and region are converted into numerical dummy variables, while `charges` is used as the target variable. The dataset is divided into **70% training and 30% testing data**, resulting in 935 training records and 402 testing records.

Three **Linear Regression models** are developed using progressively larger feature sets:

* **Model 1:** Age only
* **Model 2:** Age + BMI
* **Model 3:** All available features

**Statsmodels** is used to build and analyze the regression models, providing statistical information about the relationship between the input features and insurance charges.

The age-only model records a test **MSE of approximately 154.52 million** and an **R² of 0.097**, showing that age alone has limited predictive capability. Model performance improves when additional relevant variables are included.

Overall, the analysis demonstrates that **insurance charges depend on multiple demographic, lifestyle and health-related factors rather than age alone**. It also highlights the importance of **feature selection, categorical encoding and statistical analysis using Statsmodels** in regression modeling.

