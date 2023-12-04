from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_forest(data, dependant_variable):
    # Random forest
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, dependant_variable, test_size=0.3, random_state=42)
    x_train.to_csv("xtrain.csv")
    x_test.to_csv("xtest.csv")
    y_train.to_csv("ytrain.csv")
    y_test.to_csv("ytest.csv")

    # Apply SMOTE
    smote = SMOTE(random_state=42)

    # Create a RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)

    # Define the grid of hyperparameters to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Setup the grid search
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    #  Best parameters and best score
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Create a new RandomForestClassifier with the best parameters
    best_clf = RandomForestClassifier(**best_params, random_state=42)
    best_clf.fit(x_train, y_train)

    # Predictions and Evaluation using the fitted model with best parameters
    y_pred = best_clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display the model's performance
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Feature Importance from the fitted model
    feature_importances = pd.Series(best_clf.feature_importances_, index=data.columns).sort_values(ascending=False)
    print("Feature Importances:")
    print(feature_importances)

    # Get probabilities for each class
    y_probs = best_clf.predict_proba(x_test)

    # Set a new threshold
    threshold = 0.4  # You can tune this value

    # Apply threshold to positive class probabilities to make new predictions
    y_pred_new_threshold = (y_probs[:, 1] > threshold).astype(int)

    # Evaluate with the new predictions
    new_accuracy = accuracy_score(y_test, y_pred_new_threshold)
    new_report = classification_report(y_test, y_pred_new_threshold)

    print(f"New Accuracy: {new_accuracy}")
    print("New Classification Report:")
    print(new_report)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC Curve and AUC
    # Replace 'gnb' with 'random_forest_classifier'
    fpr, tpr, thresholds = roc_curve(y_test, best_clf.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Print AUC
    print("AUC-ROC:", roc_auc)

    # Define k-fold cross-validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    # Lists to store results of each fold
    accuracies = []
    reports = []

    for train_index, test_index in kf.split(data):
        # Split data
        x_train_kf, x_test_kf = data.iloc[train_index], data.iloc[test_index]
        y_train_kf, y_test_kf = dependant_variable.iloc[train_index], dependant_variable.iloc[test_index]

        # Apply SMOTE
        x_train_smote, y_train_smote = smote.fit_resample(x_train_kf, y_train_kf)

        # Train the model
        best_clf.fit(x_train_smote, y_train_smote)

        # Make predictions
        y_pred_kf = best_clf.predict(x_test_kf)

        # Store results
        accuracies.append(accuracy_score(y_test_kf, y_pred_kf))
        reports.append(classification_report(y_test_kf, y_pred_kf))

    # Calculate average performance metrics
    average_accuracy = np.mean(accuracies)
    print(f"Average Accuracy: {average_accuracy}")
    for i, report in enumerate(reports, 1):
        print(f"Report for fold {i}:\n{report}\n")
