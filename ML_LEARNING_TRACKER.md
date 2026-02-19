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

**Project Name:** Gold Price Prediction with RandomForestRegressor

**Project Description:** Build a REGRESSION model using RandomForest to predict gold prices based on financial market indicators. This is your second regression project with a different tree-based ensemble algorithm!

**Project Goals:**
- Learn RandomForestRegressor algorithm
- Work with financial/time-series data
- Understand correlation between gold and other financial assets
- Compare RandomForest vs XGBoost performance
- Handle date columns in datasets

**Dataset:** Gold Price Data (CSV file)
- **2,290 samples** (daily financial data from 2008-2018)
- **5 features:** Date, SPX (S&P 500), USO (Oil), SLV (Silver), EUR/USD (Currency)
- **Target:** GLD (Gold price)

**Algorithm:** RandomForest Regressor
**Files:** `4. Gold Price Prediction\0. Gold Price Prediction.ipynb`, `4. Gold Price Prediction\gld_price_data.csv`

---

## Task Breakdown

### Completed Tasks
None yet - Project just started!

### Current Task
**Task 1:** Data Loading and Initial Exploration
- Import libraries (pandas, numpy, matplotlib, seaborn, sklearn, RandomForestRegressor)
- Load CSV data
- Display basic info (shape, columns, dtypes, head)
- Check for missing values
- Understand what each feature represents

### Upcoming Tasks
**Task 2:** Data Visualization & Correlation Analysis
- Visualize target distribution
- Analyze correlations between features and gold price
- Create relevant plots

**Task 3:** Feature Engineering & Data Preparation
- Handle Date column (drop or extract features)
- Separate features and target
- Train-test split

**Task 4:** Model Training (RandomForest)
- Initialize RandomForestRegressor
- Train the model
- Make predictions

**Task 5:** Model Evaluation
- Calculate R², MAE, RMSE
- Compare with XGBoost from Project 3

**Task 6:** Model Analysis & Wrap-up
- Analyze results
- Discuss RandomForest vs XGBoost differences

---

## Learning Notes & Feedback

### Key Learnings
- **Always load CSV without headers when data has none**: Using `header=None` in `pd.read_csv()` prevents pandas from treating first row as column names
- **Class balance matters**: A 53%-47% split is fairly balanced; severe imbalance (90%-10%) would require special handling
- **Data quality check is crucial**: Checking for missing values early prevents errors during modeling
- **Feature scaling observation**: Sonar features are already in 0-1 range, suggesting pre-normalized data
- **Regression vs Classification metrics**: Classification uses accuracy/precision/recall; Regression uses R²/MAE/RMSE
- **R² interpretation**: 0.83 means model explains 83% of variance; closer to 1.0 is better
- **Overfitting detection**: Compare training vs test metrics; large gaps indicate overfitting
- **XGBoost tends to overfit**: Tree-based models can memorize training data; regularization helps

### Common Mistakes to Avoid
- **Reference vs Copy**: Using `df2 = df1` creates a reference, not a copy; use `.copy()` to create independent dataframes
- **Implicit encoding**: `.astype('category').cat.codes` encodes alphabetically - always verify which label maps to which number
- **Inconsistent naming**: Stick to conventions (lowercase `y` for target in sklearn) - **this mistake happened in both Project 1 and Project 3!**
- **Typos in variable names**: Always double-check spelling - **RECURRING ISSUE**: `traning` (Projects 1, 2), `trest` (Project 3). These typos keep appearing!
- **Copy-paste errors**: When copying code blocks, always update variable-specific parts (labels, titles, variable names) to match the new context

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

✅ **Project 3: California Housing Price Prediction with XGBoost** (COMPLETED)
- Algorithm: XGBoost Regressor (Extreme Gradient Boosting)
- Dataset: 20,640 samples, 8 features, regression
- Final Performance: Test R² = 0.8301, MAE = 0.310 ($31k), RMSE = 0.472 ($47k)
- Key Achievement: First regression project, learned XGBoost, evaluated with R²/MAE/RMSE
- Skills Learned: Regression vs classification, correlation analysis, overfitting detection, tree-based ensembles
- Notable: Model showed overfitting (training R² 0.9446 vs test 0.8301), strong predictive power (83% variance explained)
- Date Completed: 2026-02-18

---

## Next Steps
1. Student to describe the first lab project
2. Teacher will break it down into tasks
3. Begin implementation and review cycle

---

*Last Updated:* 2026-02-18
*Current Status:* Project 4 - Task 1 (Data Loading and Initial Exploration)
