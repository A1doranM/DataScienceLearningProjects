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

**Project Name:** Sonar Rock vs Mine Classification

**Project Description:** Build a binary classification model using Logistic Regression to predict whether sonar readings indicate a Rock (R) or Mine (M). The dataset contains 60 numerical features representing sonar signal frequencies/amplitudes.

**Project Goals:**
- Build a production-ready binary classification pipeline
- Learn proper data preprocessing and exploration techniques
- Implement Logistic Regression for classification
- Evaluate model performance with appropriate metrics
- Understand feature importance and model behavior

**Dataset:** `sonar_data.csv` (60 features + 1 label column)
**Algorithm:** Logistic Regression
**Files:** `1. Sonar Data Rock vs Mine\Rock vs Mine.ipynb`

---

## Task Breakdown

### Completed Tasks
✅ **Task 1:** Data Loading and Initial Exploration (COMPLETED)
- Loaded dataset correctly with proper parameters
- Explored shape, data types, and structure
- Identified 208 samples, 60 features, fairly balanced classes
- Confirmed no missing values
- Generated statistical summaries

✅ **Task 2:** Data Preprocessing (COMPLETED)
- Encoded labels successfully (M=0, R=1)
- Separated features (X: 208x60) and target (Y: 208)
- Verified numeric encoding and shapes
- Data ready for modeling

### Current Task
**Task 3:** Train-Test Split
- Split data into training and testing sets (80-20 or 70-30)
- Verify split proportions
- Check class distribution in both sets

**Task 4:** Model Training
- Initialize Logistic Regression model
- Train the model on training data
- Document training process

**Task 5:** Model Evaluation
- Predict on both training and test sets
- Calculate accuracy scores
- Generate confusion matrix
- Create classification report (precision, recall, F1-score)

**Task 6:** Model Analysis & Insights
- Analyze model performance
- Identify any issues (overfitting/underfitting)
- Discuss results and potential improvements

**Task 7:** Code Review & Best Practices
- Review entire code for production readiness
- Add necessary comments and documentation
- Final cleanup

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

### Best Practices Discovered
- (Will be populated through the projects)

---

## Project Archive

### Completed Projects
None yet

---

## Next Steps
1. Student to describe the first lab project
2. Teacher will break it down into tasks
3. Begin implementation and review cycle

---

*Last Updated:* 2026-02-12
*Current Status:* Ready to begin - awaiting project description
