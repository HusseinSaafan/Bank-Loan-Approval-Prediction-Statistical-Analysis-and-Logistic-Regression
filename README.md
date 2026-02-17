# Bank-Loan-Approval-Prediction-Statistical-Analysis-and-Logistic-Regression
A data science project that analyzes key factors influencing loan approval decisions and builds an interpretable logistic regression model. The project includes data cleaning, exploratory data analysis, statistical hypothesis testing, feature engineering, and business-oriented model interpretation.

---

> ## Disclaimer
>
> This dataset represents a **sample of a broader population** and is used for skills demonstration purposes.
>
> This project is intended to showcase the following skills:
> - Exploratory Data Analysis (EDA)
> - Statistical hypothesis testing
> - Data cleaning and preprocessing
> - Feature engineering
> - Logistic regression modeling
> - Statistical inference and coefficient interpretation
> - Building preprocessing pipelines using `scikit-learn`
> - Applying statistical modeling with `statsmodels`
> - Translating analytical results into business insights
>
> This project is **not intended for real-world banking or production deployment**.
>
> In practical financial risk modeling, the approach would be significantly more complex and would typically include:
> - Larger and more diverse datasets
> - Advanced feature engineering
> - Cross-validation and robust evaluation strategies
> - Multiple model comparison (e.g., tree-based models, ensemble methods, gradient boosting)
> - Hyperparameter tuning
> - Fairness and bias assessment
> - Regulatory compliance considerations
> - Model monitoring and performance tracking
>
> The primary focus here is interpretability, statistical validation, and demonstrating core analytical competencies rather than optimizing predictive performance for production use.

---

## Project Objective

The goal of this project is to:

- Analyze factors influencing loan approval decisions  
- Perform statistical hypothesis testing on applicant features  
- Build a predictive model to classify loan approval status  
- Interpret model coefficients and business impact  

---

## Dataset Overview

The dataset contains information about loan applicants and whether their loan was approved.

### Key Features:

| Feature | Description |
|----------|-------------|
| `Loan_ID` | Unique loan identifier |
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Education level |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant income |
| `CoapplicantIncome` | Co-applicant income |
| `LoanAmount` | Requested loan amount |
| `Loan_Amount_Term` | Loan duration |
| `Credit_History` | Credit history record |
| `Property_Area` | Urban/Semiurban/Rural |
| `Loan_Status` | Target variable (Approved/Not Approved) |

---

## Project Workflow

### 1. Data Cleaning
- Handled missing values  
- Removed invalid entries  
- Converted variables to proper types   

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis  
- Correlation checks  
- Group comparisons  
- Visualization using:
  - `matplotlib`
  - `seaborn`

### 3. Statistical Testing
Applied statistical methods to validate relationships:

- Shapiro Test – Normality testing  
- Chi-Square Test – Categorical relationships  
- T-Test – Group mean comparison  
- Proportion Z-Test – Approval rate differences  

### 4. Feature Engineering & Preprocessing
Used `ColumnTransformer` to build a clean preprocessing pipeline:

- MinMaxScaler for numerical features  
- OneHotEncoder (drop='first') for categorical features  
- Added constant term for logistic regression  

### 5. Model Building
Implemented Logistic Regression using `statsmodels`:

- Interpretable
