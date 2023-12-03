from feature_selection import feature_selection_rfe_cv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


# Function for doing a Naive Bayes classification on a chosen dataset.
# Receives input of the data frame and independent variable
# Returns nothing but does give print out of the metrics as well as the ROC curve as a graph
def naive_bayes(df, dependant_variable):
    # FEATURE SELECTION

    # Feature selection using RFE
    selected_features = feature_selection_rfe_cv(df, dependant_variable, n_features_to_select=5)

    x_selected = df[selected_features]

    # TRAINING
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_selected, dependant_variable, test_size=0.3, random_state=42)

    # Save training data
    x_train.to_csv('x_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)

    # Save testing data
    x_test.to_csv('x_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # Apply SMOTE for imbalance and oversampling
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
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, gnb.predict_proba(x_test)[:, 1])
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
