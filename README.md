# Bank-Loan-Approval-Prediction-Statistical-Analysis-and-Logistic-Regression
A data science project that analyzes key factors influencing loan approval decisions and builds an interpretable logistic regression model. The project includes data cleaning, exploratory data analysis, statistical hypothesis testing, feature engineering, and business-oriented model interpretation.

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
