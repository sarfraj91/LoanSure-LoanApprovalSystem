# LoanSure: Loan Approval System

LoanSure is a machine-learning prototype for reviewing loan applications. It uses historical applicant data to predict whether an application is likely to be **approved** or **rejected**, helping loan staff prioritize review before a human makes the final decision.

The project was prepared from the problem statement in [`Data/ProblemStatement.jpg`](Data/ProblemStatement.jpg) and the feature guide in [`Data/DataFeatures.jpg`](Data/DataFeatures.jpg). The Streamlit app is branded **LoanSure**.

## Problem and objective

The problem statement describes SecureTrust Bank, which receives personal and home loan applications from urban and rural customers. Manual checks of income, employment, credit history, and supporting information take time and can be inconsistent. The system's objective is to learn patterns from previous applications and provide a quick, consistent approval/rejection prediction to support human review.

This is a demonstration and decision-support tool. It does not replace human verification or a bank's regulated credit decision process.

## Project contents

| Path | Purpose |
| --- | --- |
| `Data/loan_approval_data.csv` | Raw dataset: 1,000 rows and 20 columns, including the approval label. |
| `Data/processed_data.csv` | Imputed and encoded dataset used by the model notebooks: 1,000 rows and 28 columns. |
| `Data/ProblemStatement.jpg` | Project brief and business objective. |
| `Data/DataFeatures.jpg` | Feature descriptions supplied with the project. |
| `dataAnalysis.ipynb` | Initial data inspection, missing-value handling, exploratory analysis, encoding, and processed-data export. |
| `models/knnModel.ipynb` | K-Nearest Neighbors experiments. |
| `models/logisticModel.ipynb` | Logistic Regression experiments. |
| `models/naiveBaisModel.ipynb` | Gaussian Naive Bayes experiments. |
| `models/randomForestModel.ipynb` | Random Forest baseline, hyperparameter search, evaluation, and model export. |
| `streamlit_app/app.py` | User-facing loan prediction form and results page. |
| `streamlit_app/loan_approval_model.pkl` | Fitted Random Forest artifact loaded by the app. |

## Dataset and features

Each row represents an applicant. The source contains these fields:

| Field | Meaning |
| --- | --- |
| `Applicant_ID` | Applicant identifier; removed before model training. |
| `Applicant_Income`, `Coapplicant_Income` | Applicant and coapplicant income. |
| `Employment_Status` | Employment category. |
| `Age`, `Marital_Status`, `Dependents`, `Gender`, `Education_Level` | Applicant profile. |
| `Credit_Score`, `Existing_Loans`, `DTI_Ratio` | Credit and current debt information. |
| `Savings`, `Collateral_Value` | Savings and pledged collateral value. |
| `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Property_Area` | Requested loan and property details. |
| `Employer_Category` | Employer group. |
| `Loan_Approved` | Target label: `Yes` means approved and `No` means rejected. |

In the processed data, the target is label-encoded (`1` = approved, `0` = rejected). Categorical predictors are one-hot encoded with a reference category dropped. The Random Forest experiment also creates `DTI_Ratio_Squre`, `Credit_Score_Square`, and `Applicant_Income_Log`; it removes the corresponding raw DTI, credit score, and applicant income columns from its training features.

## Data preparation and modeling process

The notebooks follow these steps:

1. **Load and inspect data.** `dataAnalysis.ipynb` reads the raw CSV, reviews types and summary statistics, checks missing values, and explores target and feature distributions.
2. **Handle missing values.** Numeric columns are filled with their means; categorical columns, including the target, are filled with their most frequent values.
3. **Remove the identifier.** `Applicant_ID` is excluded as it is not intended to describe creditworthiness.
4. **Encode categories.** `Loan_Approved` and `Education_Level` are label-encoded. Employment, marital status, loan purpose, property area, gender, and employer category are one-hot encoded.
5. **Create processed data.** The result is written to `Data/processed_data.csv` and reused by the model notebooks.
6. **Engineer model features.** The final model experiments add squared DTI and credit-score values and a log-transformed applicant income. The raw values for those three features are dropped in the engineered experiments.
7. **Split and train.** The final reported experiments use an 80/20 train/test split with `random_state=42`. KNN, Logistic Regression, and Gaussian Naive Bayes standardize features using a scaler fitted on the training split. Random Forest is trained on the engineered feature values without scaling.
8. **Compare models.** The notebooks report approval-class precision, recall, F1, accuracy, and a confusion matrix.
9. **Tune and save Random Forest.** `GridSearchCV` evaluates 360 parameter combinations with five-fold cross-validation, optimizing F1. The fitted best estimator is saved to `streamlit_app/loan_approval_model.pkl`.
10. **Predict in the app.** Streamlit collects applicant details, applies the same engineered fields and feature order as the Random Forest notebook, then displays an approval/rejection prediction and the model's approval probability when available.

## Model results

These are the reported results from the feature-engineered notebook runs. Metrics are percentages for the positive class (`approved = 1`). Confusion matrices use `[[TN, FP], [FN, TP]]`.

| Model | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| K-Nearest Neighbors (5 neighbors) | 76.5% | 62.96% | 55.74% | 59.13% | `[[119, 20], [27, 34]]` |
| Logistic Regression | 87.0% | 77.78% | 80.33% | 79.03% | `[[125, 14], [12, 49]]` |
| Gaussian Naive Bayes | 86.5% | 80.36% | 73.77% | 76.92% | `[[128, 11], [16, 45]]` |
| Random Forest baseline (100 trees) | 90.5% | 83.87% | 85.25% | 84.55% | `[[129, 10], [9, 52]]` |
| **Tuned Random Forest (selected model)** | **91.0%** | **84.13%** | **86.89%** | **85.48%** | **`[[129, 10], [8, 53]]`** |

The tuned Random Forest has the strongest reported accuracy and F1 score among these runs, so it is the model used by the Streamlit app. The selected estimator recorded in the notebook is a Random Forest with 300 trees and maximum depth 15 (other estimator settings remain at their defaults).

### How the classifiers make predictions

- **K-Nearest Neighbors (KNN):** finds the five most similar training applicants and predicts the class most common among them. Feature scaling is important because KNN compares distances.
- **Logistic Regression:** estimates the probability of approval from a weighted combination of the applicant features, then assigns the approved or rejected class.
- **Gaussian Naive Bayes:** estimates each class probability using the observed feature values and a Gaussian assumption for numeric features, then selects the more likely class.
- **Random Forest:** combines votes from many decision trees. The selected tuned forest uses 300 trees; the app reports the predicted class and, when available, the forest's estimated probability for approval.

Across these classifiers, `0` maps to **Rejected** and `1` maps to **Approved**. The Streamlit app loads the Random Forest artifact only; the other notebooks are model comparisons and do not provide selectable models in the app.

## Run the application

Install Python and the packages used by the app and notebooks: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `jupyter`, `matplotlib`, and `seaborn`.

From the project root, run:

```bash
streamlit run streamlit_app/app.py
```

The app asks for applicant profile, employment, financial, and loan-request information. Use the same financial units as the dataset. Enter debt-to-income as a decimal from 0 to 1 (for example, `0.30` means 30%). Numeric fields have input bounds, and the form checks that income and loan amount are greater than zero before prediction.

To retrain and export the selected model, run the cells in `models/randomForestModel.ipynb` from the `models` working directory. Its relative paths expect that directory: the notebook reads `../Data/processed_data.csv` and writes the model to `../streamlit_app/loan_approval_model.pkl`. Restart Streamlit after replacing the model so its cached estimator is refreshed.

## Important data and implementation notes

- The raw CSV has 1,000 rows, with 50 missing values in every column. This includes 50 missing approval labels. The current preprocessing fills missing target values with the most frequent label (`No`), so the processed target contains 702 rejected and 298 approved records. Imputing unknown outcomes as rejected can bias training and evaluation; results should be treated as provisional until this is reviewed.
- Imputation and encoding are performed before the train/test split in the current analysis notebook. This allows information from the eventual test set to influence preprocessing. A production-quality evaluation should fit all preprocessing only on each training fold, ideally in a scikit-learn pipeline, and should exclude rows with unknown targets rather than inventing labels.
- The written feature guide and problem statement do not fully match the raw CSV categories. For example, the raw data contains `Contract` employment and `Business` loan-purpose/employer categories; the Streamlit form does not currently offer all of these categories. Align the form, source data, and training categories before relying on predictions for those applicant types.
- The source brief describes an Indian banking context, while the app currently formats requested loan amount with a `$` symbol. The CSV does not explicitly document a currency, so confirm and standardize the currency display and data units.
- Some earlier notebook cells use different split settings or print accuracy under a `Precision Score` label. The results table above uses the final feature-engineered 80/20 run outputs; the model notebooks should be kept consistent and rerun from top to bottom when results change.
- The current app is a prototype. Historical metrics do not guarantee future performance, fairness, or suitability for individual credit decisions.
