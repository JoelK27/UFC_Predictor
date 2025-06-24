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
├── data/                        # Raw data (CSV) and generated files
│
├── models/                      # Trained model files (created automatically)
│
├── src/                         # Python source code (preprocessing, training, analysis, prediction)
│   ├── data_exploration.py      # Data exploration and cleaning
│   ├── data_preprocessing.py    # Data preprocessing (scaling, encoding)
│   ├── model_training.py        # Model training and feature analysis
│   ├── analysis.py              # (Optional) Feature importance analysis/plotting
│   └── predict_fight.py         # Script for interactive fight prediction
│
├── main.py                      # Central workflow script (step-by-step, fully documented)
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

The workflow is fully linear and documented in `main.py`:

- **Data Exploration & Cleaning:**  
  Loads the raw data, checks for missing values, explores the structure, and removes unnecessary columns.  
  Saves a cleaned version for further processing.

- **Preprocessing:**  
  Scales numeric features, encodes the target variable, and extracts additional features such as weight class and gender.  
  Saves the preprocessed data for modeling.

- **Model Training & Analysis:**  
  Trains a separate Random Forest model for each weight class and gender.  
  Evaluates model performance, analyzes feature importance, and saves the trained models for later use.  
  Visualizes the top 10 most important features for each model.

- **Interpretation:**  
  Short conclusions and findings are printed in the script output after each step.

---

## Predicting a Fight

After running `main.py` and training the models, you can predict the outcome of a specific fight using the interactive script:

```bash
python src/predict_fight.py
```

You will be prompted to enter:
- Fighter 1 name
- Fighter 2 name
- Weight class (e.g. "Flyweight")
- Gender ("Men" or "Women")

The script will output the predicted winner and the 10 most important features for the selected model.

---

## Notes

- **Python scripts only:**  
  The entire project runs as `.py` files, no notebooks.
- **All steps are documented** and commented so the workflow is easy to follow and reproducible.
- **All analyses and findings** are included directly in the code and console output.
- **Models are saved automatically** in the `models/` folder after training.

---

## Contact

For questions or feedback, please contact the Github! (JoelK27/UFC_Predictor)
