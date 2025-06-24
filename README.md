# UFC_Predictor

Semester project for AI and Data Science.  
A classification model to determine the winner and analyze key factors for UFC fights using data accumulated between July 2016 and November 2024.

---

## Project Overview

This project builds a machine learning model to predict the outcome of UFC fights and analyzes the most important influencing factors.  
The entire workflow is organized as Python scripts and is fully reproducible.

---

## Project Structure

```
UFC_Predictor/
│
├── data/                        # Raw data (CSV)
│
├── src/                         # Python source code (preprocessing, training, analysis)
│
├── main.py                      # Central workflow script (step-by-step)
├── environment.yml              # Anaconda environment for reproducibility
└── README.md                    # This file
```

---

## Reproducibility & Setup

1. **Create Anaconda Environment**  
   Make sure [Anaconda](https://www.anaconda.com/products/individual) is installed.  
   Then run in the project folder:
   ```bash
   conda env create -f environment.yml
   conda activate ufc-predictor
   ```

2. **Data**  
   The raw data file (`UFC Fight Statistics (July 2016 - Nov 2024).csv`) should be placed in the `data/` folder.  
   If not present, please download it from the official source or your course platform and place it there.

3. **Run the Project**  
   Start the entire workflow with:
   ```bash
   python main.py
   ```

---

## Workflow (main.py)

- **Data Exploration & Cleaning:**  
  Removes empty/unnecessary columns, handles missing values, checks the target variable.
- **Preprocessing:**  
  Scales numeric features, encodes the target variable.
- **Model Training:**  
  Trains a separate model for each weight class and gender.
- **Analysis:**  
  Displays the most important factors (feature importances) for fight outcomes.
- **Interpretation:**  
  Short conclusions and findings are printed in the script output.

---

## Notes

- **Python scripts only:**  
  The entire project runs as `.py` files, no notebooks.
- **All steps are documented** and commented so the workflow is easy to follow.
- **Own analyses and findings** are included directly in the code and console output.

---