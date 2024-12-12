# Lend or Lose - Loan Default Prediction Project

## Overview
This project implements machine learning models to predict loan defaults based on borrower characteristics and loan attributes. Our solution employs multiple classification algorithms to achieve robust prediction performance.

## Models Implemented

### Deep Learning Models (`NN_ML.ipynb`)
- Neural Network with multiple fully connected layers
- Batch normalization and dropout regularization
- PyTorch implementation

### SVM and Logistic Regression (`Logistic_Regression.ipynb` and `SVM_ML_Project.ipynb`)
- Support Vector Machines (SVM)
- Logistic Regression with different regularization:
  - L1 (Lasso)
  - L2 (Ridge)
  - ElasticNet

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
| Neural Network | 88.746% | One-Hot Encoding, Batch Normalization, Dropout |
| Gradient Boosting Classifier | 88.707% | One-Hot Encoding and Hyperparameter Tuning |
| Logistic Regression (L1) | 88.580% | StandardScaling, One-Hot Encoding |
| Gaussian Naive Bayes | 88.488% | One-Hot Encoding |
| SVM | 88.370% | MinMaxScaling, One-Hot Encoding |
| Bernoulli Naive Bayes | 88.447% | One-Hot Encoding |
| KNN Classifier | 88.447% | Normalization and Hyperparameter Tuning |
| Decision Trees | 80.072% | One-Hot Encoding and Hyperparameter Tuning |
| Multinomial Naive Bayes | 57.918% | One-Hot Encoding |

Key findings:
- XGBoost achieved the best performance with a score of 0.88752
- Linear models showed competitive performance with proper regularization
- Neural Network demonstrated robust performance with proper architecture
- Tree-based models consistently performed well
- Feature engineering and hyperparameter tuning were crucial for optimal results

## Project Structure
```
├── NN_ML.ipynb                  # Neural Network implementation
├── SVM_ML_Project.ipynb         # SVM implementation
├── Logistic_Regression.ipynb    # Logistic Regression implementation
├── TreeBasedModels.ipynb        # Tree-based implementations
├── KNN_Bayes_Classifier.py      # KNN and Bayes implementations
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
└── README.md
```

## Requirements
- Python 3.x
- pandas
- scikit-learn
- xgboost
- pytorch
- jupyter (for notebook execution)

You can install the required packages using:
```bash
pip install pandas scikit-learn xgboost torch jupyter
```

## How to Run

### For Neural Network Model
1. Launch Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `NN_ML.ipynb`
3. Run all cells in sequence

### For Linear Models
1. Launch Jupyter Notebook
2. Open `Logistic_Regression.ipynb` and `SVM_ML_Project.ipynb`
3. Run all cells in sequence

### For Tree-Based Models
1. Launch Jupyter Notebook:
bash
jupyter notebook

2. Open TreeBasedModels.ipynb
3. Run all cells in sequence to:
   - Load and preprocess data
   - Train models
   - Generate predictions
   - Create submission file
### For KNN and Bayes Models
1. Run the Python script directly:
bash
python KNN_Bayes_Classifier.py

## Data Description
The models use various features including:
- Borrower demographics (Age, Education, Employment)
- Financial indicators (Income, Credit Score)
- Loan characteristics (Amount, Term, Interest Rate)
- Other relevant factors (Marital Status, Dependencies)

## Preprocessing Techniques
- One-Hot Encoding for categorical variables
- StandardScaler for neural networks and linear models
- MinMaxScaling for SVM
- Normalization for KNN implementation
- Feature selection based on correlation analysis
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Batch normalization for neural networks
- Dropout regularization for preventing overfitting

For more information regarding preprocessing and other details, kindly refer the ML_Report.pdf file

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.
## License
[MIT](https://choosealicense.com/licenses/mit/)
