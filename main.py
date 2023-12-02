import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from data_cleaning import clean_data
from feature_selection import feature_selection_rfe
from feature_selection import feature_selection_rfe_cv
from feature_selection import feature_selection_with_cv
from feature_selection import remove_multicollinearity_vars
from visuals import check_multicollinearity
from imblearn.over_sampling import SMOTE


df = pd.read_csv("churn_clean.csv")

x_reference, x_analysis, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list, df_analysis = clean_data(df)
dependant_variable = x_analysis['Churn']
print("Data Cleaning and Transformation Complete")

# Selecting only the numerical columns
numerical_cols = df_analysis[continuous_list]
#
# Apply the remove_multicollinearity_vars function
reduced_numerical_cols = remove_multicollinearity_vars(numerical_cols)

# You can then combine these reduced numerical columns back with the categorical ones if needed
final_df = pd.concat([reduced_numerical_cols, x_analysis.select_dtypes(include=['object'])], axis=1)

# FEATURE SELECTION

# Feature selection using RFE
selected_features = feature_selection_rfe_cv(final_df, y, n_features_to_select=5)
# Feature selection with decision tree
estimator = DecisionTreeRegressor(random_state=42)

# x_selected, selected_features, fitted_estimator = feature_selection_with_cv(final_df, y, estimator)
x_selected = final_df[selected_features]
# print(len(x_selected), len(y))

# TRAINING
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the model on the training set with SMOTE
gnb.fit(x_train_smote, y_train_smote)

# Make predictions
y_pred = gnb.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
