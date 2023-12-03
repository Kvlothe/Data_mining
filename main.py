import pandas as pd
from random_forest import random_forest
from data_cleaning import clean_data
from feature_selection import remove_multicollinearity_vars


# Read in the data and create a date frame - Need to edit this to have the use pick what data they want to use
df = pd.read_csv("churn_clean.csv")

# Clean data - Input the newly read in data
# Returns columns that are dropped from analysis, columns for analysis, one-hot and binary categorical columns,
# continuous columns and lists of both
x_reference, x_analysis, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list, df_analysis = clean_data(df)

# Would like to make this a users choice - For now it is hard-coded
dependant_variable = x_analysis['Churn']
print("Data Cleaning and Transformation Complete")

# Selecting only the numerical columns
numerical_cols = x_analysis[continuous_list]

# Ensure all columns are of numeric type
numerical_cols = numerical_cols.apply(pd.to_numeric, errors='coerce')

# Recheck data types
print(numerical_cols.dtypes)

# Recheck for any remaining missing values
print(numerical_cols.isnull().sum())

# Apply the remove_multicollinearity_vars function
reduced_numerical_cols = remove_multicollinearity_vars(numerical_cols)

# You can then combine these reduced numerical columns back with the categorical ones if needed
final_df = pd.concat([reduced_numerical_cols, x_analysis.select_dtypes(include=['object'])], axis=1)

random_forest(final_df, dependant_variable)
