# UFC_Predictor – Project Presentation

---

## 1. Introduction

- **Goal:** Build a machine learning pipeline to predict the winner, fight outcome (finish or decision), and number of rounds for UFC fights.
- **Data:** UFC Fight Statistics from July 2016 to November 2024.
- **Motivation:** Can we predict fight results and understand the key factors that decide a UFC bout?

---

## 2. Data Overview

- **Source:** Official UFC fight statistics (CSV)
- **Scope:** 8+ years, all major weight classes, men & women
- **Features:** Fighter stats (strikes, takedowns, age, reach, etc.), bout info, outcome
- **Target Variables:** Winner, outcome type, rounds

*Visualization:*
- Show a table or chart with sample features and target columns.
- Pie chart: Distribution of fight outcomes (finish vs. decision).

---

## 3. Data Exploration & Cleaning

- Checked for missing values and outliers.
- Removed irrelevant or empty columns.
- Explored distributions of key features.

*Visualization:*
- Histogram: Age or reach distribution.
- Bar chart: Most common weight classes.

---

## 4. Feature Engineering & Preprocessing

- **Scaling:** Standardized numeric features.
- **Encoding:** Encoded categorical variables (winner, outcome, rounds).
- **Extraction:** Parsed weight class and gender from bout info.

*Visualization:*
- Correlation heatmap of main numeric features.
- Example: Before/after scaling plot.

---

## 5. Modeling Approach

- **Model:** Random Forest Classifiers/Regressors
- **Separate models:** For each weight class and gender
- **Targets:** Winner, outcome, rounds

*Visualization:*
- Diagram: Workflow from raw data → preprocessing → modeling → prediction.

---

## 6. Model Training & Evaluation

- Trained models on preprocessed data.
- Evaluated using accuracy (classification) and MAE (regression).
- Analyzed feature importance for each model.

*Visualization:*
- Bar chart: Model accuracy per weight class.
- Line/bar: MAE for rounds prediction.

---

## 7. Feature Importance

- Identified top 10 features for each prediction task.
- Example: Strikes landed, takedown defense, age, reach.

*Visualization:*
- Horizontal bar chart: Top 10 features for winner prediction.
- Compare with outcome/rounds feature importances.

---

## 8. Fight Prediction Demo

- **Interactive script:** User enters fighter names, weight class, gender.
- **Outputs:** Predicted winner, outcome, rounds, and key features.

*Visualization:*
- Screenshot or mockup of prediction script in action.
- Example prediction result.

---

## 9. Key Insights & Findings

- Certain features (e.g., significant strikes, takedown defense) are highly predictive.
- Model performance varies by weight class and gender.
- Predicting rounds is more challenging than winner/outcome.

*Visualization:*
- Table: Summary of best/worst performing models.
- Highlight surprising findings.

---

## 10. Conclusion & Outlook

- **Summary:** Built a robust, reproducible pipeline for UFC fight prediction.
- **Limitations:** Data quality, feature availability, unpredictability of sports.
- **Possible Future Work:** Add more features (e.g., fighter form, injuries), try deep learning, deploy as web app.

*Visualization:*
- Flowchart: Potential future improvements.

---

**(Export slides as PDF for submission!!)**