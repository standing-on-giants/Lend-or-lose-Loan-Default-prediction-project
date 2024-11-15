import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
train_df = pd.read_csv("./train_ml_project.csv")
test_df = pd.read_csv("./test_ml_project.csv")

undropped_test_df = test_df

# Dropping the LoanID column since it shouldn't be used for making predictions
train_df = train_df.drop(['LoanID'], axis=1)
test_df = test_df.drop(['LoanID'], axis=1)

# Identify categorical and numerical columns
categorical_columns_train = train_df.select_dtypes(include=['object']).columns
numerical_columns_train = train_df.select_dtypes(include=['int64', 'float64']).columns

categorical_columns_test = test_df.select_dtypes(include=['object']).columns
numerical_columns_test = test_df.select_dtypes(include=['int64', 'float64']).columns

# Initialize the encoder
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform only the categorical columns for train data
categorical_encoded_train = encoder.fit_transform(train_df[categorical_columns_train])

# Transform the categorical columns for test data using the same encoder
categorical_encoded_test = encoder.transform(test_df[categorical_columns_train])

# Create DataFrame with encoded categorical variables
encoded_categorical_traindf = pd.DataFrame(
    categorical_encoded_train,
    columns=encoder.get_feature_names_out(categorical_columns_train)
)
encoded_categorical_testdf = pd.DataFrame(
    categorical_encoded_test,
    columns=encoder.get_feature_names_out(categorical_columns_train)
)

# Combine with numerical columns (ensure we don't include 'Default' in test_df)
encoded_train_df = pd.concat([
    encoded_categorical_traindf,
    train_df[numerical_columns_train].reset_index(drop=True)
], axis=1)

encoded_test_df = pd.concat([
    encoded_categorical_testdf,
    test_df[numerical_columns_test].reset_index(drop=True)
], axis=1)

# Separate features and target variable for training
X = encoded_train_df.drop('Default', axis=1)
y = train_df['Default']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the scaler
scaler = MinMaxScaler()

# Scale the training and testing features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the models
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_test_pred_gnb = gnb.predict(X_test)
gnb_test_accuracy = accuracy_score(y_test, y_test_pred_gnb)
print(f"Gaussian Naive Bayes Accuracy: {gnb_test_accuracy}")

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_test_pred_bnb = bnb.predict(X_test)
bnb_test_accuracy = accuracy_score(y_test, y_test_pred_bnb)
print(f"Bernoulli Naive Bayes Accuracy: {bnb_test_accuracy}")


mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_test_pred_mnb = mnb.predict(X_test)
mnb_test_accuracy = accuracy_score(y_test, y_test_pred_mnb)
print(f"Multinomial Naive Bayes Accuracy: {mnb_test_accuracy}")


# KNN parameters for grid search
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn = KNeighborsClassifier()

# Perform grid search with scaled data
knn_grid_search = GridSearchCV(knn, knn_params, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
knn_grid_search.fit(X_train_scaled, y_train)

# Evaluate the best KNN model on the scaled test set
best_knn = knn_grid_search.best_estimator_
y_test_pred_knn = best_knn.predict(X_test_scaled)
knn_test_accuracy = accuracy_score(y_test, y_test_pred_knn)
print(f"K-Nearest Neighbors Accuracy (with normalization): {knn_test_accuracy}")

# Select the best model based on test accuracy
# best_model_name = max(
#     [('GaussianNB', gnb_test_accuracy, gnb),
#      ('BernoulliNB', bnb_test_accuracy, bnb),
#      ('MultinomialNB', mnb_test_accuracy, mnb),
#      ('KNN', knn_test_accuracy, best_knn)],
#     key=lambda x: x[1]
# )

# best_model, best_test_accuracy, best_model_object = best_model_name

# Print the best model and its accuracy
# print(f"Best Model: {best_model} with Accuracy: {best_test_accuracy}")

# Use the best model to predict on the entire test dataset
best_predictions = mnb.predict(encoded_test_df)

def generate_submission(predictions, df):
    submission = pd.DataFrame({'LoanID': df['LoanID'], 'Default': predictions})
    submission.to_csv("LendOrLose_submission.csv", index=False)

# Generate the submission file
generate_submission(best_predictions, undropped_test_df)
print("Submission file created successfully!")
