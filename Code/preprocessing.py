from typing import Tuple, Optional

from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, \
    precision_recall_curve, average_precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Input: application data path, credit data path, encode type (OneHot/Label)
# Output: features train, features test, target train, target test
def dataPreprocessing(application_data_path: str,
                      credit_data_path: str,
                      encode_type: str
                      ) -> Optional[Tuple[DataFrame, DataFrame, DataFrame, DataFrame]]:
    # Read CSV file
    application_data = pd.read_csv(application_data_path)
    credit_data = pd.read_csv(credit_data_path)

    print("Length of application_data", len(application_data))
    print("Length of credit_data", len(credit_data))

    # Get common id in two csv files
    common_customers = set(application_data['ID']).intersection(set(credit_data['ID']))

    print("Common customers ID: \n", common_customers)
    print("Numbers of common customers: \n", len(common_customers))

    # Only keep the data of the id that two csv files have in common
    application_data = application_data[application_data['ID'].isin(common_customers)]
    credit_data = credit_data[credit_data['ID'].isin(common_customers)]

    print('Valid application data: \n', application_data)
    print('Valid credit data: \n', credit_data)

    # List all columns with NA values
    application_data_columns_with_null_values = application_data.columns[application_data.isnull().any()].tolist()
    credit_data_columns_with_null_values = credit_data.columns[credit_data.isnull().any()].tolist()

    print("Application data columns with null values: \n",
          application_data_columns_with_null_values)  # Only 'OCCUPATION_TYPE' column has null value
    print("Credit data columns with null values: \n", credit_data_columns_with_null_values)  # No null value columns

    # Sort the application data with id & occupation_type
    application_data_columns_to_sort = ['ID', 'OCCUPATION_TYPE']

    # Put the rows with null values at last
    ascending_list = [True, False]
    application_data = application_data.sort_values(by=application_data_columns_to_sort, ascending=ascending_list)

    print('Application data after sorting: \n', application_data)

    # Look into the duplicated data
    duplicate_rows = application_data[application_data.duplicated(subset='ID', keep=False)]
    print('Duplicated rows in application data: \n', duplicate_rows)

    # Duplicate, delete all duplicated data, as same id does not necessarily correspond to the same person
    application_data = application_data.drop_duplicates(subset='ID', keep=False)

    print('Application data after duplication: \n', application_data)

    # Drop irrelevant columns
    application_data = application_data.drop(columns=['CODE_GENDER'])

    print("Application data after dropping irrelevant columns: \n", application_data)

    # Remove outliers
    # A person on file should not be younger than 18 years old (6570 days)
    # A person should not be employed for more than 60 years (21900 days)
    # A person should not have more children than family members
    # A person's working years should not be bigger than age
    condition = (application_data['DAYS_BIRTH'] >= -6570) | \
                (application_data['CNT_FAM_MEMBERS'] <= application_data['CNT_CHILDREN']) | \
                (application_data['DAYS_EMPLOYED'] < -21900) | \
                (application_data['DAYS_EMPLOYED'] < application_data['DAYS_BIRTH'])

    application_data = application_data[~condition]

    print("Application data after removing outliers: \n", application_data)

    # How many records of DAYS_EMPLOYED bigger than 0
    days_employed_gt_0_count = (application_data['DAYS_EMPLOYED'] > 0).sum()
    print("Sum of days employed bigger than 0: \n", days_employed_gt_0_count)

    # Process the employment data, DAYS_EMPLOYED bigger than 0 means unemployed, correct it to be 0, or drop it
    #application_data['DAYS_EMPLOYED'] = application_data['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else x)
    application_data.drop(application_data[application_data['DAYS_EMPLOYED'] >= 0].index, inplace=True)
    print("After removing data of DAYS_EMPLOYED bigger than 0: \n", application_data)

    # Fill the OCCUPATION_TYPE with NA value
    application_data['OCCUPATION_TYPE'].fillna('Not Collected', inplace=True)

    # List the data type of all the features
    application_data_types = application_data.dtypes

    print("Application data types: \n", application_data_types)

    # Transfer Binary Feature from Y/N to 1/0
    columns_to_convert = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    application_data[columns_to_convert] = application_data[columns_to_convert].replace({'Y': 1, 'N': 0})

    # Observe the numeric features distribution
    # Number of children
    counts_1, bins_1, _ = plt.hist(application_data['CNT_CHILDREN'], bins=20, edgecolor='black')
    plt.xlabel('Number of children')
    plt.ylabel('Frequency')
    plt.title('Number of children Distribution')
    plt.show()
    print("Number of children Distribution: \n")
    for i in range(len(bins_1) - 1):
        print(f"Bin {i + 1}: {bins_1[i]} - {bins_1[i + 1]}--{counts_1[i]}--{counts_1[i] / (len(application_data))}")

    # Annual income
    counts_2, bins_2, _ = plt.hist(application_data['AMT_INCOME_TOTAL'], bins=20, edgecolor='black')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.title('Income Distribution')
    plt.show()
    print("Annual income Distribution: \n")
    for i in range(len(bins_2) - 1):
        print(f"Bin {i + 1}: {bins_2[i]} - {bins_2[i + 1]}--{counts_2[i]}--{counts_2[i]/(len(application_data))}")

    # Age
    counts_3, bins_3, _ = plt.hist((application_data['DAYS_BIRTH'] / (-365)).astype(float), bins=20, edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')
    plt.show()
    print("Age Distribution: \n")
    for i in range(len(bins_3) - 1):
        print(f"Bin {i + 1}: {bins_3[i]} - {bins_3[i + 1]}--{counts_3[i]}--{counts_3[i] / (len(application_data))}")

    # Working years
    counts_4, bins_4, _ = plt.hist((application_data['DAYS_EMPLOYED'] / (-365)).astype(float), bins=20, edgecolor='black')
    plt.xlabel('Working years')
    plt.ylabel('Frequency')
    plt.title('Working years Distribution')
    plt.show()
    print("Working years Distribution: \n")
    for i in range(len(bins_4) - 1):
        print(f"Bin {i + 1}: {bins_4[i]} - {bins_4[i + 1]}--{counts_4[i]}--{counts_4[i] / (len(application_data))}")

    # Family size
    counts_5, bins_5, _ = plt.hist(application_data['CNT_FAM_MEMBERS'], bins=20, edgecolor='black')
    plt.xlabel('Family size')
    plt.ylabel('Frequency')
    plt.title('Family size Distribution')
    plt.show()
    print("Family size Distribution: \n")
    for i in range(len(bins_5) - 1):
        print(f"Bin {i + 1}: {bins_5[i]} - {bins_5[i + 1]}--{counts_5[i]}--{counts_5[i] / (len(application_data))}")

    # Tranform numeric features into category
    # Number of children
    print('CNT_CHILDREN_BEFORE_LABELED:', application_data['CNT_CHILDREN'])

    def categorize_number_of_children(number_of_children):
        if number_of_children == 0:
            return 'No-child'
        elif number_of_children == 1:
            return 'One-child'
        elif number_of_children == 2:
            return 'Two-children'
        else:
            return 'Three-or-more-children'

    application_data['CNT_CHILDREN'] = application_data['CNT_CHILDREN'].apply(lambda x: categorize_number_of_children(x))

    print('CNT_CHILDREN_AFTER_LABELED:', application_data['CNT_CHILDREN'])

    # Annual income
    print('AMT_INCOME_TOTAL_BEFORE_LABELED:', application_data['AMT_INCOME_TOTAL'])

    def categorize_annual_income(annual_income):
        if annual_income < 100000:
            return 'Low-income'
        elif 100000 <= annual_income < 200000:
            return 'Low-to-middle-income'
        elif 200000 < annual_income <= 500000:
            return 'Middle-income'
        elif 500000 < annual_income < 1000000:
            return 'Middle-to-high-income'
        else:
            return 'High-income'

    application_data['AMT_INCOME_TOTAL'] = application_data['AMT_INCOME_TOTAL'].apply(lambda x: categorize_annual_income(x))

    print('AMT_INCOME_TOTAL_AFTER_LABELED:', application_data['AMT_INCOME_TOTAL'])

    # Age
    application_data['DAYS_BIRTH'] = (application_data['DAYS_BIRTH'] / (-365)).astype(float)
    print('DAYS_BIRTH_BEFORE_LABELED:', application_data[['ID', 'DAYS_BIRTH']])

    def categorize_age(age):

        if age < 18:
            return 'Under-18'
        elif 18 <= age < 26:
            return '18-25'
        elif 26 <= age < 36:
            return '26-35'
        elif 36 <= age < 46:
            return '36-45'
        elif 46 <= age < 56:
            return '46-55'
        else:
            return 'Above-55'

    application_data['DAYS_BIRTH'] = application_data['DAYS_BIRTH'].apply(lambda x: categorize_age(x))

    print('DAYS_BIRTH_AFTER_LABELED:', application_data[['ID', 'DAYS_BIRTH']])

    # Working years
    application_data['DAYS_EMPLOYED'] = (application_data['DAYS_EMPLOYED'] / (-365)).astype(float)
    print('DAYS_EMPLOYED_BEFORE_LABELED:', application_data['DAYS_EMPLOYED'])

    def categorize_working_years(working_years):

        if working_years < 1:
            return 'Less-than-a-year'
        elif 1 <= working_years < 3:
            return 'One-to-three-years'
        elif 3 <= working_years < 5:
            return 'Three-to-five-years'
        elif 5 <= working_years < 10:
            return 'Five-to-ten-years'
        else:
            return 'Above-ten-years'

    application_data['DAYS_EMPLOYED'] = application_data['DAYS_EMPLOYED'].apply(lambda x: categorize_working_years(x))

    print('DAYS_EMPLOYED_AFTER_LABELED:', application_data['DAYS_EMPLOYED'])

    # Family size
    print('CNT_FAM_MEMBERS_BEFORE_LABELED:', application_data['CNT_FAM_MEMBERS'])

    def categorize_family_size(family_size):
        if family_size == 1:
            return 'One-member'
        elif family_size == 2:
            return 'Two-members'
        elif family_size == 3:
            return 'Three-members'
        else:
            return 'Four-and-more-members'

    application_data['CNT_FAM_MEMBERS'] = application_data['CNT_FAM_MEMBERS'].apply(lambda x: categorize_family_size(x))

    print('CNT_FAM_MEMBERS_AFTER_LABELED:', application_data['CNT_FAM_MEMBERS'])

    # List the data type of all the features after transferring
    application_data_types = application_data.dtypes

    print("Application data types after transferring: \n", application_data_types)

    # Extract feature of object type
    features_of_numeric_type = application_data.select_dtypes(include=['float64', 'int64'])
    features_of_object_type = application_data.select_dtypes(exclude=['float64', 'int64'])

    # Encode features of object type
    if encode_type == "OneHot":
        # One Hot Encode
        # one_hot_encoder = OneHotEncoder(sparse_output=False)
        # one_hot_encoded_features = one_hot_encoder.fit_transform(features_of_object_type)
        one_hot_encoded_features = pd.get_dummies(features_of_object_type,
                                                  columns=features_of_object_type.columns.tolist(),
                                                  dummy_na=False, dtype=int)
        application_data = pd.concat([features_of_numeric_type, pd.DataFrame(one_hot_encoded_features)],
                                     axis=1)
        # features_after_one_hot_encoded = features_after_one_hot_encoded.astype(float)

        print("features_after_one_hot_encoded: \n", application_data)
        application_data.to_csv("../features_after_one_hot_encoded.csv", index=False)

    else:
        # Label Encode
        label_encoder = LabelEncoder()
        label_encoded_features = features_of_object_type.apply(label_encoder.fit_transform)
        application_data = pd.concat([features_of_numeric_type, pd.DataFrame(label_encoded_features)], axis=1)

        print("features_after_label_encoded: \n", application_data)
        application_data.to_csv("../features_after_label_encoded.csv", index=False)

    null_columns = application_data.columns[application_data.isnull().any()].tolist()
    if len(null_columns) > 0:
        null_columns_str = ', '.join(map(str, null_columns))
        print(f"Columns with null value: {null_columns}")
    else:
        print("No column with null value")

    # Check if the status of customers in credit_record.csv contains 1/2/3/4/5
    bad_customers = credit_data[(credit_data['STATUS'].isin(['1', '2', '3', '4', '5']))]['ID'].unique()

    print("Bad customers: \n", bad_customers)
    print("Bad customer numbers: \n", len(bad_customers))

    # Add the label column to application data & output csv
    application_data['IsGoodOrBad'] = application_data['ID'].isin(bad_customers).map({True: 0, False: 1})

    print('Bad customers in application: \n', application_data[application_data['ID'].isin(bad_customers)])
    print('application data with label: \n', application_data)

    application_data.to_csv('../application_labeled.csv', index=False)

    # Drop ID, ID has nothing to do with features
    application_data = application_data.drop(columns=['ID'])

    print("Application data after dropping irrelevant columns: \n", application_data)

    # Randomly divide the dataset into train & test
    application_train, application_test, customer_label_train, customer_label_test = \
        (train_test_split(application_data.drop('IsGoodOrBad', axis=1),
                          application_data['IsGoodOrBad'], test_size=0.2, random_state=42))

    df = pd.DataFrame(customer_label_train)

    def label_to_category(label):
        if label == 0:
            return 'Bad'
        else:
            return 'Good'

    df['Category'] = df['IsGoodOrBad'].apply(label_to_category)
    label_counts = df['Category'].value_counts().values
    converted_labels = df['Category'].value_counts().index

    print("Labels Distribution:", label_counts)

    # Draw pie chart
    plt.figure(figsize=(8, 4))
    plt.pie(label_counts, labels=converted_labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.axis('equal')
    plt.title('Distribution of Customer Labels')
    plt.show()

    # Using Synthetic Minority Over-Sampling Technique(SMOTE) to overcome sample imbalance problem
    # smote = SMOTE(random_state=42)
    # Using (Adaptive Synthetic Sampling) ADASYN to overcome sample imbalance problem
    adasyn = ADASYN()
    # application_train_smote, customer_label_train_smote = smote.fit_resample(application_train, customer_label_train)
    application_train_smote, customer_label_train_smote = adasyn.fit_resample(application_train, customer_label_train)

    return application_train_smote, application_test, customer_label_train_smote, customer_label_test

if __name__ == "__main__":

    # Test code
    model_type = input("Input the Model type (1-5):")
    application_train, application_test, customer_label_train, customer_label_test = (
        dataPreprocessing('../Data/application_record.csv',
                      '../Data/credit_record.csv', 'OneHot'))

    print("application_train: \n", application_train)
    print("application_test: \n", application_test)
    print("customer_label_train: \n", customer_label_train)
    print("customer_label_test: \n", customer_label_test)

    if model_type == '1':
        model = LogisticRegression(C=0.8,
                                   random_state=0,
                                   solver='lbfgs')
    elif model_type == '2':
        model = DecisionTreeClassifier(max_depth=12,
                                       min_samples_split=8,
                                       random_state=1024)

    elif model_type == '3':
        model = RandomForestClassifier(n_estimators=250,
                                       max_depth=12,
                                       min_samples_leaf=16
                                       )

    elif model_type == '4':
        model = SVC(C=100, kernel='linear')
        train_size = int(0.01 * len(application_train))
        application_train = application_train[:train_size]
        customer_label_train = customer_label_train[:train_size]

    elif model_type == '5':
        model = XGBClassifier(max_depth=12,
                              n_estimators=250,
                              min_child_weight=8,
                              subsample=0.8,
                              learning_rate=0.02,
                              seed=42)
    else:
        model = CatBoostClassifier(iterations=250,
                                   learning_rate=0.2,
                                   od_type='Iter',
                                   verbose=25,
                                   depth=16,
                                   random_seed=42)

    model.fit(application_train, customer_label_train)

    customer_label_predict = model.predict(application_test)

    # Accuracy
    accuracy = accuracy_score(customer_label_test, customer_label_predict)
    print("Accuracy:", accuracy)

    # Precision
    precision = precision_score(customer_label_test, customer_label_predict)
    print("Precision:", precision)

    # Recall rate
    recall = recall_score(customer_label_test, customer_label_predict)
    print("Recall:", recall)

    # F1 score
    f1 = f1_score(customer_label_test, customer_label_predict)
    print("F1 Score:", f1)

    # AUC-ROC
    roc_auc = roc_auc_score(customer_label_test, customer_label_predict)
    print("AUC-ROC:", roc_auc)

    # AUC-PR
    precision_1, recall_1, _ = precision_recall_curve(customer_label_test, customer_label_predict)
    pr_auc = auc(recall_1, precision_1)
    print("AUC-PR:", pr_auc)

    # Plot ROC line
    fpr, tpr, _ = roc_curve(customer_label_test, customer_label_predict)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot PR line
    precision, recall, _ = precision_recall_curve(customer_label_test, customer_label_predict)
    plt.figure(figsize=(8, 8))
    plt.step(recall, precision, color='b', where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(customer_label_test, customer_label_predict)
    print("Confusion Matrix:")
    print(conf_matrix)

