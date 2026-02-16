# ML & Data Science Learning Journey - Project Tracker

## Learning Objectives
- Gain production-ready experience in building ML models
- Work on modern AI/ML tasks
- Develop best practices for data science workflows
- Build portfolio of complete ML projects

## Background
- ✅ Math background
- ✅ Completed Machine Learning Specialization on Coursera
- 🎯 Now focusing on: Production-ready model building and modern ML tasks

## Teaching Approach
1. Student describes lab project
2. Teacher breaks it into manageable tasks
3. Student implements code for each task
4. Teacher reviews code, identifies mistakes, provides corrections
5. Iterate until task is complete
6. Move to next task

---

## Current Project
**Status:** ✅ COMPLETED

**Project Name:** Diabetes Prediction using SVM

**Project Description:** Build a binary classification model using Support Vector Machine (SVM) to predict whether a patient has diabetes based on diagnostic measurements. The Pima Indians Diabetes dataset contains medical predictor variables and one target variable (Outcome).

**Project Goals:**
- Learn Support Vector Machine (SVM) algorithm
- Handle medical/health data with proper preprocessing
- Deal with potential data quality issues (zero values)
- Compare SVM performance with previous Logistic Regression knowledge
- Build production-ready medical prediction model

**Dataset:** `diabetes.csv` (8 features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age + 1 Outcome)
**Algorithm:** Support Vector Machine (SVM)
**Files:** `2. Diabetes Prediction\Diabetes Prediction using Machine Learning.ipynb`

---

## Task Breakdown

### Completed Tasks
✅ **Task 1:** Data Loading and Initial Exploration (COMPLETED)
- Imported all required libraries correctly
- Loaded 768-patient diabetes dataset with proper headers
- Identified 8 features + 1 target (Outcome)
- Found class imbalance: 65% non-diabetic, 35% diabetic
- Discovered critical data quality issue: 5-374 zeros in medical features (missing data)
- No null values but zeros represent missing measurements

✅ **Task 2:** Data Quality & Preprocessing (COMPLETED)
- Replaced 0 with NaN for medically impossible features (Glucose, BP, SkinThickness, Insulin, BMI)
- Imputed 652 missing values total using median imputation
- Separated features (X: 768×8) and target (y: 768)
- Applied StandardScaler for SVM compatibility
- All features now standardized (mean=0, std=1)

✅ **Task 3:** Train-Test Split (COMPLETED)
- Split data 80-20 with stratification
- Training: 614 samples, Test: 154 samples
- Class balance maintained: ~65% class 0, ~35% class 1 in both sets
- Verified shapes and distributions

✅ **Task 4:** Model Training (SVM) (COMPLETED)
- Initialized SVC with linear kernel
- Trained on 614 samples successfully
- Model uses 315 support vectors (51% of training data)
- Class 0: 157 SVs, Class 1: 158 SVs

✅ **Task 5:** Model Evaluation (COMPLETED)
- Predictions on train and test sets
- Training: 78.01%, Test: 76.62%
- Generated confusion matrix and classification report
- Better generalization than Logistic Regression (1.39% vs 14.71% drop)
- BONUS: Built predictive system for new patient data

### Project Complete! 🎉
All core tasks finished. Ready for next project.

---

## Learning Notes & Feedback

### Key Learnings
- **Always load CSV without headers when data has none**: Using `header=None` in `pd.read_csv()` prevents pandas from treating first row as column names
- **Class balance matters**: A 53%-47% split is fairly balanced; severe imbalance (90%-10%) would require special handling
- **Data quality check is crucial**: Checking for missing values early prevents errors during modeling
- **Feature scaling observation**: Sonar features are already in 0-1 range, suggesting pre-normalized data

### Common Mistakes to Avoid
- **Reference vs Copy**: Using `df2 = df1` creates a reference, not a copy; use `.copy()` to create independent dataframes
- **Implicit encoding**: `.astype('category').cat.codes` encodes alphabetically - always verify which label maps to which number
- **Inconsistent naming**: Stick to conventions (lowercase `y` for target in sklearn)
- **Typos in variable names**: Always double-check spelling - typos like `traning` vs `training` affect code quality and readability

### Best Practices Discovered
- **Use `.copy()` for dataframe operations**: Prevents unintended modifications to original data
- **Explicit label encoding with `.map()`**: More readable and maintainable than implicit methods
- **Follow sklearn conventions**: Use lowercase `y` for target, uppercase `X` for features
- **Always verify transformations**: Print value_counts() after encoding to confirm mapping

---

## Project Archive

### Completed Projects
✅ **Project 1: Sonar Rock vs Mine Classification** (COMPLETED)
- Algorithm: Logistic Regression
- Dataset: 208 samples, 60 features, binary classification
- Final Performance: 71.43% test accuracy, 86.14% training accuracy
- Key Achievement: Built complete ML pipeline from data loading to evaluation
- Skills Learned: Data preprocessing, train-test split, model evaluation, confusion matrix analysis
- Date Completed: 2026-02-14

✅ **Project 2: Diabetes Prediction using SVM** (COMPLETED)
- Algorithm: Support Vector Machine (Linear Kernel)
- Dataset: 768 samples, 8 features, binary classification (imbalanced)
- Final Performance: 76.62% test accuracy, 78.01% training accuracy
- Key Achievement: Handled missing data, applied feature scaling, built predictive system
- Skills Learned: Data quality checks, median imputation, StandardScaler, SVM, medical data analysis
- Notable: Excellent generalization (only 1.39% drop), 315 support vectors
- Date Completed: 2026-02-16

---

## Next Steps
1. Student to describe the first lab project
2. Teacher will break it down into tasks
3. Begin implementation and review cycle

---

*Last Updated:* 2026-02-12
*Current Status:* Ready to begin - awaiting project description
