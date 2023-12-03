import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


# Function for Stepwise Selection - foreword adding and backward elimination
# Inputs are independent variables and the dependant variable
# Returns the selected features, function will stop adding features if they do not improve the model
def stepwise_selection(independent_variables, dependent_variable, initial_list=[],
                       threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(independent_variables.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(dependent_variable, sm.add_constant(pd.DataFrame(independent_variables[included +
                                                                                                  [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # Backward step
        model = sm.OLS(dependent_variable, sm.add_constant(pd.DataFrame(independent_variables[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break

    return included


# Function for removing multicollinearity variables
# Input is the independent variables and will return the data frame with variables removed
# that were causing multicollinearity
def remove_multicollinearity_vars(data, threshold=10):
    high_vif = True
    while high_vif:
        # Calculate VIFs
        vif_data = pd.DataFrame()
        vif_data["feature"] = data.columns
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

        # Check for variables with high VIF
        high_vif_vars = vif_data[vif_data['VIF'] > threshold]
        if high_vif_vars.empty:
            high_vif = False
        else:
            # Remove the variable with the highest VIF
            highest_vif_var = high_vif_vars.sort_values('VIF', ascending=False).iloc[0]
            data = data.drop(highest_vif_var['feature'], axis=1)
            print(f"Removed {highest_vif_var['feature']} with VIF: {highest_vif_var['VIF']}")

    return data


def decision_tree(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # Initialize the DecisionTreeClassifier
    dt = DecisionTreeRegressor(random_state=42)

    # Fit the model
    dt.fit(x_train, y_train)

    # Get feature importances
    importances = dt.feature_importances_

    # Transform the importances into a readable format
    feature_importance = zip(x.columns, importances)
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Print the feature importance
    for feature, importance in feature_importance:
        print(f"Feature: {feature}, Importance: {importance}")

    # Calculate the average feature importance
    average_importance = np.mean(importances)

    # Select features that have importance greater than the average
    selected_features = [feature for feature, importance in feature_importance if importance > average_importance]

    print("Selected features based on importance threshold:")
    print(selected_features)
    print()
    # Create the design matrix X and target vector y using only selected features
    x_selected = x[selected_features]

    return x[selected_features], selected_features, dt


# Feature selection with cross-validation
def feature_selection_with_cv(x, y, estimator, cv=5):
    # Perform cross-validation
    scores = cross_val_score(estimator, x, y, cv=cv, scoring='r2')
    print(f"Cross-validation R^2 scores: {scores}")
    print(f"Average R^2 score: {np.mean(scores)}")

    # Compute feature importances on the full dataset
    # and select features based on the importances from the full model
    estimator_clone = clone(estimator)
    estimator_clone.fit(x, y)
    importances = estimator_clone.feature_importances_

    # Transform the importances into a readable format
    feature_importance = zip(x.columns, importances)
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Print the feature importance
    for feature, importance in feature_importance:
        print(f"Feature: {feature}, Importance: {importance}")

    # Select features based on a more discriminating threshold, like the 75th percentile
    threshold = np.percentile(importances, 75)
    selected_features = [feature for feature, importance in feature_importance if importance > threshold]

    print("Selected features based on importance threshold:")
    for feature in selected_features:
        print(feature)

    # Perform cross-validation again, but only with the selected features
    selected_x = x[selected_features]
    scores_selected = cross_val_score(estimator, selected_x, y, cv=cv, scoring='r2')
    print(f"Cross-validation R^2 scores with selected features: {scores_selected}")
    print(f"Average R^2 score with selected features: {np.mean(scores_selected)}")

    return selected_x, selected_features, estimator_clone


def feature_selection_rfe(x, y, n_features_to_select=8):
    # Scaling the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Create a logistic regression classifier with increased max_iter
    logreg = LogisticRegression(max_iter=1000)

    # RFE model
    rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(x_train, y_train)

    # Summarize the selection of the attributes
    print('Selected features:', list(x.columns[rfe.support_]))
    print('Feature Ranking:', rfe.ranking_)

    return x.columns[rfe.support_]


def feature_selection_rfe_cv(x, y, n_features_to_select=5, cv_folds=5):
    # Scaling the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Create a logistic regression classifier with increased max_iter
    logreg = LogisticRegression(max_iter=1000)

    # RFE with cross-validation
    rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(cv_folds),
                  scoring='accuracy', min_features_to_select=n_features_to_select)
    rfecv.fit(x_scaled, y)

    # Summarize the selection of the attributes
    print('Optimal number of features:', rfecv.n_features_)
    print('Selected features:', list(x.columns[rfecv.support_]))
    print('Feature Ranking:', rfecv.ranking_)

    return x.columns[rfecv.support_]
