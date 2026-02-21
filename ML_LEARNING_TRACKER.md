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

**Project Name:** Customer Segmentation with K-Means Clustering

**Project Description:** Build an UNSUPERVISED LEARNING model using K-Means Clustering to segment mall customers into groups based on their income and spending patterns. This is your first unsupervised learning project - NO target variable, NO train-test split!

**Project Goals:**
- Understand unsupervised learning (no labels/target)
- Learn K-Means Clustering algorithm
- Use the Elbow Method to find optimal number of clusters
- Visualize customer segments with colored scatter plots
- Extract business insights from cluster analysis

**Dataset:** Mall Customers Data (CSV file)
- **200 samples** - mall customer data
- **Features:** CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
- **Target:** NONE - unsupervised learning!
- **Key insight:** Group customers by Annual Income and Spending Score

**Algorithm:** K-Means Clustering
**Files:** `5. Customer Segmentation\0. Customer Segmentation.ipynb`, `5. Customer Segmentation\mall_customers.csv`

---

## Task Breakdown

### Completed Tasks
None yet - Project just started!

### Current Task
**Task 1:** Data Loading and Initial Exploration

### Upcoming Tasks
**Task 2:** Finding Optimal Clusters (Elbow Method)
- Calculate WCSS for k=1 to 10
- Plot the elbow curve
- Identify optimal k

**Task 3:** Train K-Means Model
- Initialize KMeans with optimal k
- Fit to data
- Get cluster labels

**Task 4:** Visualize Customer Segments
- Scatter plot with colored clusters
- Mark cluster centroids
- Interpret business meaning of each segment

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
- **RandomForest vs XGBoost**: RandomForest builds trees in parallel (averaging predictions), XGBoost builds sequentially (correcting errors)
- **RandomForest generalizes better**: On gold price dataset, RandomForest showed only 1% R² drop vs XGBoost's 11.4% drop
- **Feature correlation importance**: Strong predictor (Silver 0.867 correlation) led to 98.85% test R² on gold prices
- **KISS Principle**: Simple model (98.85% R²) outperformed over-engineered v3.0 model (3.77% R²) - complexity ≠ performance
- **Returns vs Absolute prices**: Predicting % changes is harder for tree models than absolute prices in low-noise datasets
- **Model extrapolation failure**: ML models cannot predict beyond training range - always check data distribution before testing
- **Time-based splits**: For time-series data, always split chronologically to avoid future data leakage

### Common Mistakes to Avoid
- **Reference vs Copy**: Using `df2 = df1` creates a reference, not a copy; use `.copy()` to create independent dataframes
- **Implicit encoding**: `.astype('category').cat.codes` encodes alphabetically - always verify which label maps to which number
- **Inconsistent naming**: Stick to conventions (lowercase `y` for target in sklearn) - **this mistake happened in both Project 1 and Project 3!**
- **Typos in variable names**: Always double-check spelling - **RECURRING ISSUE**: `traning` (Projects 1, 2), `trest` (Project 3). These typos keep appearing!
- **Copy-paste errors**: When copying code blocks, always update variable-specific parts (labels, titles, variable names) to match the new context
- **Scatter plot axes confusion**: Always put feature on x-axis and target on y-axis to show cause → effect relationship correctly

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

✅ **Project 4: Gold Price Prediction with RandomForest** (COMPLETED)
- Algorithm: RandomForestRegressor (parallel tree ensemble)
- Dataset: 2,290 samples, 4 features (SPX, USO, SLV, EUR/USD), financial time-series data
- Final Performance: Test R² = 0.9885, MAE = 1.352 ($135), RMSE = 2.468 ($247)
- Key Achievement: Outstanding 98.85% R² with minimal overfitting (only 1% drop)
- Skills Learned: Financial data analysis, RandomForest algorithm, comparing tree-based ensembles, handling date columns
- Notable: Much better generalization than XGBoost (1% vs 11.4% R² drop), strong feature (Silver 0.867 correlation) enabled excellent predictions
- Date Completed: 2026-02-18

---

## Next Steps
1. Student to describe the first lab project
2. Teacher will break it down into tasks
3. Begin implementation and review cycle

---

*Last Updated:* 2026-02-21
*Current Status:* Project 5 - Task 1 (Customer Segmentation - Data Loading)
