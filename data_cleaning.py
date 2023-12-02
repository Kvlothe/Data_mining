import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.impute import SimpleImputer
from visuals import plot_outliers
import logging


# Configure logging
logging.basicConfig(filename='data_cleaning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Attempt to cap outliers - did not improve model, new approach needed
def cap_outliers(series, lower_percentile=5, upper_percentile=95):
    lower_limit = series.quantile(lower_percentile / 100)
    upper_limit = series.quantile(upper_percentile / 100)
    return series.clip(lower=lower_limit, upper=upper_limit)


# Method for checking each column in the data frame and imputing the values based on standard imputation procedures
# Input is the data frame, output is the data frame with no missing values
def impute_missing_values(df):
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()

    # Impute missing values in categorical columns with the mode
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # Impute missing values in numerical columns with the mean
    imputer_numerical = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    return df


# Method for checking the data frame for duplicate values/columns and removing them
# Input is data frame and output is data frame with duplicates removed
def remove_duplicates(df):
    # Check for duplicates and remove them
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        df = df.drop_duplicates()
        # Use logging
        logging.info(f'Removed {num_duplicates} duplicate rows.')
    else:
        print('No duplicates found.')

    return df


# Check the data frame for outliers and plotting the outliers as well as printing out the Z-score and IQR
# Input is the data frame for checking and the output is a folder of Box-plots of the outliers
def outliers(df):
    # Outliers
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    # Initializing a list to store outlier info
    outliers_info_list = []

    for col in numeric_cols.columns:
        data = numeric_cols[col]

        # Z-Score Method
        z_score = np.abs((data - data.mean()) / data.std())
        outliers_z_score = np.sum(z_score > 3)  # Typically, a Z-Score above 3 is considered as an outlier

        # IQR Method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers_iqr = np.sum((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))  # IQR Rule for Outliers

        # Storing info
        outliers_info_list.append({'Column': col,
                                   'Num_Outliers_ZScore': outliers_z_score,
                                   'Num_Outliers_IQR': outliers_iqr})

    # Creating DataFrame
    outliers_info = pd.DataFrame(outliers_info_list)

    print(outliers_info)
    # Call the function
    plot_outliers(df)


# Method for viewing the data types in the data frame for analysis. Good way to see what data you are working with
def view_data_types(df):
    dtype_groups = df.columns.groupby(df.dtypes)
    # Print out each data type and its columns of each data type
    for dtype, columns in dtype_groups.items():
        print(f"\nData Type: {dtype}")
        for column in columns:
            print(f"- {column}")


# Method for only this data set, as it renames the survey questions with the appropriate questions from the dictionary
# Input is the data frame for analysis and returns the data frame with the columns renamed
def rename_columns(df):
    # Relabel the columns listed as item1...item8 with appropriate questions
    df.rename(columns={'Item1': 'Timely response',
                       'Item2': 'Timely fixes',
                       'Item3': 'Timely replacement',
                       'Item4': 'Reliability',
                       'Item5': 'Options',
                       'Item6': 'Respectful response',
                       'Item7': 'Courteous exchange',
                       'Item8': 'Evidence of active listening'},
              inplace=True)

    return df


# Method for going through the data frame and applying binary mapping on the data frame
# Input is the data frame and the mapping that was specified, output is the data frame with binary mapping applied
def apply_binary_mapping(df, binary_mapping):
    mapped_columns = []  # Initialize a list to store mapped columns
    for col in df.columns:
        unique_values = df[col].unique()
        if len(unique_values) == 2 and all(value in binary_mapping for value in unique_values):
            df[col] = df[col].map(binary_mapping)
            mapped_columns.append(col)  # Add the mapped column to the list
    return df, mapped_columns


# Method for going through the data frame and applying one-hot encoding on the data frame
# Input is the data frame and returns the same data frame with the one-hot encoded columns applied
def apply_one_hot_encoding(df):
    encoded_columns = []  # Initialize a list to store encoded columns
    for col in df.columns:
        if len(df[col].unique()) > 2 and isinstance(df[col].dtype, CategoricalDtype):
            df = pd.get_dummies(df, columns=[col], prefix=[col], drop_first=True)
            encoded_columns.append(col)  # Add the encoded column to the list
    return df, encoded_columns


# Method for cleaning and transforming the data. Input is the original data set/frame
# This function will call multiple functions that will clean and transform the data
# Returns columns not used for analysis, columns used for analysis after cleaning and transformation,
# categorical columns, continuous columns, one-hot encoded columns, binary columns,
# and the full data frame without the columns that are not used for analysis
# I wanted to return anything that might be of use for any type of analysis or visuals
def clean_data(data):

    # View the data types in the Data frame
    view_data_types(data)

    # Remove duplicates
    data_no_duplicates = remove_duplicates(data)

    # Impute missing values automatically
    data_imputed = impute_missing_values(data_no_duplicates)

    # View outliers
    outliers(data_imputed)

    # Rename the survey questions (Churn Dataset only)
    columns_renamed = rename_columns(data_imputed)

    # Create a group for columns that I want to keep around but do not want to use for analysis, then create
    columns_to_keep = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'Zip', 'Job', 'Population', 'Lat', 'Lng',
                       'City', 'State', 'County', 'Area', 'PaymentMethod', 'TimeZone']
    data_analysis = columns_renamed.drop(columns=columns_to_keep)

    # Encoding
    # Define your binary mapping
    binary_mapping = {'Yes': 1, 'No': 0, 'DSL': 1, 'Fiber Optic': 0}

    # Apply binary mapping to binary columns automatically and get a list of mapped columns
    data_mapped, mapped_binary_columns = apply_binary_mapping(data_analysis, binary_mapping)

    # Apply one-hot encoding to suitable columns automatically and get a list of encoded columns
    data_encoded, encoded_columns = apply_one_hot_encoding(data_mapped)

    # separate the data frames - columns to keep for later and columns for analysis
    x_reference = data[columns_to_keep]
    x_analysis = data_encoded
    df_analysis = data.drop(columns=columns_to_keep)
    x_analysis.to_csv('churn_analysis.csv')
    x_reference.to_csv('analysis_reference.csv')

    # Create categories and groups of columns - Categorical and Continuous for ease of use in graphically viewing data
    categorical_columns = encoded_columns + mapped_binary_columns

    continuous_list = [col for col in x_analysis.columns if col not in categorical_columns]
    # print(continuous_list)

    # If you need to create a list of continuous columns after encoding
    continuous_columns = [col for col in x_analysis.columns if col not in categorical_columns]
    print()

    return x_reference, x_analysis, encoded_columns, mapped_binary_columns, categorical_columns, \
        continuous_columns, continuous_list, df_analysis
