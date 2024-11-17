# Lend or Lose - Loan Default Prediction Project

## Overview
This project implements machine learning models to predict loan defaults based on borrower characteristics and loan attributes. Our solution employs multiple classification algorithms to achieve robust prediction performance.

## Models Implemented

### Tree-Based Models (`TreeBasedModels.ipynb`)
- Decision Tree Classifier
- XGBoost
- Gradient Boosting Classifier

### Alternative Models (`KNN_Bayes_Classifier.py`)
- K-Nearest Neighbors (KNN)
- Naive Bayes Classifier (Multinomial, Bernoulli, and Gaussian variants)

## Model Performance

| Model | Kaggle Score | Approach |
|-------|--------------|----------|
| XG Boost | 88.752% | One-Hot Encoding and Hyperparameter Tuning |
| Gradient Boosting Classifier | 88.707% | One-Hot Encoding and Hyperparameter Tuning |
| Gaussian Naive Bayes | 88.488% | One Hot Encoding |
| Bernoulli Naive Bayes | 88.447% | One Hot Encoding |
| KNN Classifier | 88.447% | Normalization and Hyperparameter Tuning |
| Decision Trees | 80.072% | One Hot Encoding and Hyperparameter Tuning |
| Multinomial Naive Bayes | 57.918% | One Hot Encoding |

Key findings:
- XGBoost achieved the best performance with a score of 0.88752
- Tree-based models (XGBoost and Gradient Boosting) consistently performed better
- Simple Naive Bayes models showed competitive performance
- Feature engineering and hyperparameter tuning were crucial for optimal results

## Project Structure
```
├── TreeBasedModels.ipynb        # Jupyter notebook with tree-based implementations
├── KNN_Bayes_Classifier.py      # Python script with KNN and Bayes implementations
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
└── README.md
```

## Requirements
- Python 3.x
- pandas
- scikit-learn
- xgboost
- jupyter (for notebook execution)

You can install the required packages using:
```bash
pip install pandas scikit-learn xgboost jupyter
```

## How to Run

### For Tree-Based Models
1. Launch Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `TreeBasedModels.ipynb`
3. Run all cells in sequence to:
   - Load and preprocess data
   - Train models
   - Generate predictions
   - Create submission file

### For KNN and Bayes Models
1. Run the Python script directly:
```bash
python KNN_Bayes_Classifier.py
```

## Data Description
The models use various features including:
- Borrower demographics (Age, Education, Employment)
- Financial indicators (Income, Credit Score)
- Loan characteristics (Amount, Term, Interest Rate)
- Other relevant factors (Marital Status, Dependencies)

## Preprocessing Techniques
- One-Hot Encoding for categorical variables
- Normalization for KNN implementation
- Feature selection based on correlation analysis
- Hyperparameter tuning using GridSearchCV

## Output
Both implementations generate a submission file named `LendOrLose_submission.csv` containing:
- LoanID
- Default prediction (0 or 1)

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.

## License
[MIT](https://choosealicense.com/licenses/mit/)
