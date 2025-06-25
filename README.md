# UFC_Predictor

Semester project for AI and Data Science  
A machine learning pipeline to predict the winner, fight outcome (finish or decision), and number of rounds for UFC fights, using data from July 2016 to November 2024.

---

## Project Overview

This project develops several machine learning models to predict key aspects of UFC fights:
- **Winner prediction:** Which fighter will win.
- **Outcome prediction:** Whether the fight ends by finish or decision.
- **Rounds prediction:** The expected number of rounds the fight will last.

All scripts are written in Python and the workflow is fully reproducible.

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
│   └── predict_fight.py         # Interactive script for fight prediction
│
├── main.py                      # Central workflow script (step-by-step, fully documented)
├── environment.yml              # Anaconda environment for reproducibility
└── README.md                    # This file
```

---

## Setup & Reproducibility

1. **Create Anaconda Environment**  
   Make sure [Anaconda](https://www.anaconda.com/products/individual) is installed.  
   Then run in the project folder:
   ```bash
   conda env create -f environment.yml
   conda activate ufc-predictor
   ```

2. **Data**  
   Place the raw data file (`UFC Fight Statistics (July 2016 - Nov 2024).csv`) in the `data/` folder.  
   If not present, download it from the official source or your course platform.

3. **Run the Project**  
   Start the full workflow with:
   ```bash
   python main.py
   ```

---

## Workflow

The workflow is fully linear and documented in `main.py`:

- **Data Exploration & Cleaning:**  
  Loads the raw data, checks for missing values, explores the structure, and removes unnecessary columns.  
  Saves a cleaned version for further processing.

- **Preprocessing:**  
  Scales numeric features, encodes target variables (winner, outcome, rounds), and extracts additional features such as weight class and gender.  
  Saves the preprocessed data for modeling.

- **Model Training & Analysis:**  
  Trains separate Random Forest models for winner, outcome, and rounds for each weight class and gender.  
  Evaluates model performance, analyzes feature importance, and saves the trained models for later use.  
  Visualizes the top 10 most important features for each model.

- **Interpretation:**  
  Key findings and conclusions are printed after each step in the script output.

---

## Fight Prediction

After running `main.py` and training the models, you can predict the outcome of a specific fight using the interactive script:

```bash
python src/predict_fight.py
```

You will be prompted to enter:
- Fighter 1 name
- Fighter 2 name
- Weight class (e.g. "Flyweight")
- Gender ("Men" or "Women")

The script will output:
- Predicted winner
- Predicted fight outcome (finish or decision)
- Predicted number of rounds
- The 10 most important features for the selected model

---

## Notes

- **Python scripts only:**  
  The entire project runs as `.py` files, no notebooks.
- **All steps are documented** and commented for easy reproducibility.
- **All analyses and findings** are included directly in the code and console output.
- **Models are saved automatically** in the `models/` folder after training.

---

## Contact

For questions or feedback, please contact via Github! (JoelK27/UFC_Predictor)