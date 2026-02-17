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
**Status:** 🏗️ In Progress

**Project Name:** California Housing Price Prediction with XGBoost

**Project Description:** Build a REGRESSION model using XGBoost to predict median house values in California districts. This is your first regression project - predicting continuous values instead of categories!

**Project Goals:**
- Learn XGBoost (Extreme Gradient Boosting) algorithm
- Transition from classification to regression problems
- Learn regression evaluation metrics (R², MAE, RMSE)
- Handle larger datasets (20,640 samples)
- Understand feature importance in tree-based models

**Dataset:** California Housing (built-in sklearn dataset)
- **20,640 samples** (much larger than previous projects!)
- **8 features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target:** Median house value (continuous, in $100,000s)

**Algorithm:** XGBoost Regressor
**Files:** `3. House Price Prediction\House Price Prediction.ipynb`

---

## Task Breakdown

### Completed Tasks
✅ **Task 1:** Data Loading and Initial Exploration (COMPLETED)
- Imported all required libraries including XGBoost
- Loaded California Housing dataset (20,640 samples, 8 features)
- Converted to DataFrame with proper structure
- No missing values, all float64 data types
- Target range: $15k-$500k (note: values capped at $500k)
- Understood feature meanings through DESCR

### Current Task
**Task 2:** Data Visualization & Understanding
- Import libraries (numpy, pandas, xgboost, sklearn)
- Load California Housing dataset using `fetch_california_housing()`
- Convert to DataFrame for easier handling
- Display dataset info, shape, and first few rows
- Check for missing values
- Generate statistical summary
- **NEW:** Understand target distribution (continuous values, not categories)

### Upcoming Tasks
**Task 2:** Data Visualization & Understanding
- Visualize target distribution (histogram)
- Check feature correlations
- Understand what each feature represents

**Task 3:** Train-Test Split
- Split data (NO stratification - this is regression!)
- Verify shapes

**Task 4:** Model Training (XGBoost)
- Initialize XGBoost Regressor
- Train the model
- Understand XGBoost hyperparameters

**Task 5:** Model Evaluation (Regression Metrics)
- Calculate R² score, MAE, RMSE
- Visualize predictions vs actual values
- Analyze feature importance

**Task 6:** Model Analysis & Improvements
- Discuss results
- Compare with previous projects
- Suggest improvements

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
