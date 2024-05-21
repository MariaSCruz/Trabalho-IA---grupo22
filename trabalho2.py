import numpy as np
import random
import os
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import read_table, read_csv
import seaborn as sns
from matplotlib import pyplot as plt

#Read the DataFrame
dt = pd.read_csv('hcc_dataset.csv')

#Resolution of Work 2

#Feature Assessment:

# To replace '?' by np.nan, except in the 'Encephalopathy' and 'Ascites' column, which already have null values and we want them to be interpreted that way
for x in dt.columns:
    if x != 'Encephalopathy' and x!= 'Ascites':
        dt[x] = dt[x].replace('?', np.nan)
    elif x=='Encephalopathy':
        dt[x] = dt[x].replace(np.nan, 'None')
        dt[x] = dt[x].replace('?', np.nan)
    else:
        dt[x] = dt[x].replace(np.nan, 'None')
        dt[x] = dt[x].replace('?', np.nan)

# Check column names and their data types
#print("Columns in the DataFrame and their data types:")
#print(dt.dtypes)
#serves to check that we have all objects, even when they should be float

# Identify numeric columns
number_cols = []
for x in dt.columns:
    c=0
    for i in range(dt.shape[0]):
        if not pd.isna(dt.loc[i, x]) and c==0:
            # Try to convert the value in the first line to numeric
            valor = pd.to_numeric(dt.loc[i, x], errors='coerce')
            # Check if the conversion was successful (not NaN)
            if not pd.isna(valor):
                number_cols.append(x)
            c+=1
#print(len(number_cols))

# Convert columns identified as possible float to float
potential_float_cols = []
for x in number_cols:
    if dt[x].dtype == 'object':
        dt[x] = pd.to_numeric(dt[x], errors='coerce')
        if pd.api.types.is_numeric_dtype(dt[x]):
            potential_float_cols.append(x)

# Check the columns that were identified as float
#print("\nColumns that were identified as float:")
#print(potential_float_cols)

# Update data type to float for identified columns
for col in potential_float_cols:
    dt[col] = dt[col].astype(float)

#Check if it was changed correctly
#print(dt.dtypes)

# Converting lists to sets to use set difference
columns = set(dt.columns)
columns_n= set(number_cols)

# Getting the difference between sets
complement_number_cols = list(columns - columns_n)

# Replace null values in numeric columns with the average
for col in number_cols:
    mean_value = dt[col].astype(float).mean()
    dt[col] = dt[col].fillna(mean_value)

# Replace null values in non-numeric columns with mode
for col in complement_number_cols:
    mode_value = dt[col].mode()[0]
    dt[col] = dt[col].fillna(mode_value)

# Final check: null values per column (expected to be all zeros except in the Encephalopathy column and Ascites)
#print("\nNull values per column after filling:")
#print(dt.isnull().sum())

# Display detailed DataFrame information
#print("\nDetailed DataFrame information:")
#print(dt.info())

#Dataset Dimensions

#print(dt.shape)

#Descriptive Statistics
#print("Descriptive Statistics:\n", dt.describe())

# Set output directories
output_dir_hist = r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Histogramas'
output_dir_box = r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Boxplot'
output_dir_scatter = r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Scatterplotmatrix'

# Histograms for Numeric Data
'''for column in number_cols:
    plt.figure()
    dt[column].hist()
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir_hist, f'histogram_{column}.png'))
    plt.close() 

# Box Plots for Numeric Data
for column in number_cols:
    plt.figure()
    sns.boxplot(y=dt[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.savefig(os.path.join(output_dir_box, f'boxplot_{column}.png'))
    plt.close()

# Scatter Plot Matrix for each Numeric Data
for col in number_cols:
    sns.pairplot(dt[number_cols], vars=[col])
    plt.savefig(os.path.join(output_dir_scatter, f'scatterplot_matrix_{col}.png'))
    plt.close()'''

# Scatter Plot Matrix for all Numeric Data together
'''sns.pairplot(dt[number_cols])
plt.savefig(os.path.join(output_dir_scatter, 'scatterplot_matrix.png'))
plt.close()'''

#Remove columns with low variance

# Check and delete columns according to the criteria
columns_to_drop = []

for col in dt.columns:
    value_counts = dt[col].value_counts()
    if len(value_counts) == 2 and value_counts.min() <= 20:
        columns_to_drop.append(col)

# Delete columns that meet the criteria
dt.drop(columns=columns_to_drop, inplace=True)

# Display deleted columns
#print("Dropted columns:", columns_to_drop)

#Inspecting the correlation matrix

#Pass non-numeric values to 0 and 1

# Getting only non-numeric columns that are not in the list of columns to remove
columns_not_numeric_used_now = [col for col in complement_number_cols if col not in columns_to_drop]
#print(len(columns_not_numeric_used_now))

d_01 = {}
for i in columns_not_numeric_used_now: #i - column names
    c = 0
    if i=='PS':
        d_01.update({'Active': 0,'Restricted':1,'Ambulatory':2,'Selfcare':3, 'Disabled':4})
    elif i=='Class':
        d_01.update({'Lives': 1, 'Dies': 0})
    elif i=='Ascites':
        d_01.update({'None': 1, 'Mild': 2,'Moderate/Severe':3})
    elif i=='Encephalopathy':
        d_01.update({'None': 1, 'Grade I/II': 2, 'Grade III/IV': 3})
    else:
        for j in range(dt.shape[0]): #j - total number of lines
            if j==0:#the first one doesn't compare to the previous one obviously
                for chave in {dt.loc[j, i]: c}.keys():
                    if chave not in d_01:
                        d_01.update({dt.loc[j, i]: c})
                        c+=1
            elif dt.loc[j,i]!=dt.loc[j-1,i]: #add different values if it is a different attribute
                for chave in {dt.loc[j, i]: c}.keys():
                    if chave not in d_01:
                        d_01.update({dt.loc[j, i]: c})
                        c+=1
#print(d_01) #d_01 is the dictionary in which the keys are the different types of non-numeric attributes and their value is the corresponding numeric one

# Configure pandas option
pd.set_option('future.no_silent_downcasting', True)

# Replacing the values in the DataFrame
for col in columns_not_numeric_used_now:
    dt[col] = dt[col].replace(d_01)

# Scatter Plot Matrix with Class

output_dir_scatter2 = r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Scatterplotmatrix2'

# Define a specific column for comparison
specific_column = "Class"

# Iterate over the DataFrame columns, excluding the Class column
'''for col in dt.columns:
    if col != specific_column:
        # Create the scatter plot
        plt.figure()
        sns.scatterplot(data=dt, x=specific_column, y=col)
        plt.title(f'{specific_column} vs {col}')

        # Save the chart in the specified folder
        plt.savefig(os.path.join(output_dir_scatter2, f'{specific_column}_vs_{col}.png'))
        plt.close()'''

# Create the correlation matrix

#Calculate the relationship coefficient differently when it comes to 0's and 1's for 0's and 1's
def edgeworth_correlation(x, y):
    n = len(x)
    contingency_table = np.histogram2d(x, y, bins=(np.max(x) + 1, np.max(y) + 1))[0]
    contingency_table = contingency_table / n
    contingency_table_sum = np.sum(contingency_table)

    expected_values = np.outer(np.sum(contingency_table, axis=1), np.sum(contingency_table, axis=0))

    edgeworth_correlation = np.sum((contingency_table - expected_values) ** 2 / expected_values)

    return edgeworth_correlation

#Get only numeric columns that are not in the removed columns list
number_cols2 = [col for col in number_cols if col not in columns_to_drop]
#print(len(number_cols2))

#Organize lists
columns_not_numeric_used_now_set=set(columns_not_numeric_used_now)
columns_not_numeric_used_now_no_01={'Encephalopathy','PS','Ascites'}

columns_01=list(columns_not_numeric_used_now_set-columns_not_numeric_used_now_no_01)

#Calculate the Edgeworth correlation between non-numeric columns replaced by 0's and 1's, and the Class
d_edgeworth_corr={}
for x in columns_01:
    edgeworth_corr = edgeworth_correlation(dt[x], dt['Class'])
    d_edgeworth_corr.update({x:edgeworth_corr})
#print(d_edgeworth_corr) #dictionary in which the keys are the respective columns and their values are the coefficients calculated for them

#Correlation matrix
corr_matrix = dt.corr()

#Correlation calculated by default for numerical values and those that were not just 0's and 1's (Pearson)
for y in number_cols2:
    correlation_x_class = corr_matrix.loc[y, 'Class']
    d_edgeworth_corr.update({y:correlation_x_class})
#print(d_edgeworth_corr)

for z in columns_not_numeric_used_now_no_01:
    correlation_x_class = corr_matrix.loc[z, 'Class']
    d_edgeworth_corr.update({z:correlation_x_class})
#print(d_edgeworth_corr)  # with the coefficients done this way

# Create a DataFrame with correlation coefficients
corr_df = pd.DataFrame(columns=['Class'], index=dt.columns)

# Fill the DataFrame with dictionary values
corr_df['Class'] = d_edgeworth_corr.values()

# Print the DataFrame
#print(corr_df)

# Plot the heat map (Edgeworth for non-numeric values replaced by 0's and 1's)
'''plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlation Matrix - Annotations')'''
#output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Mapas_calor_corr'
#plt.savefig(os.path.join(output_dir, f'Heat_map_class_edgeworth.png'))
#plt.close()

# Fill the DataFrame with absolute values from the dictionary
corr_df['Class'] = [abs(correlation) for correlation in d_edgeworth_corr.values()]

# Plot the heat map with abs values (Edgeworth for non-numeric values replaced by 0's and 1's)
'''plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlation Matrix - Annotations')'''
#output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Mapas_calor_corr'
#plt.savefig(os.path.join(output_dir, f'Heat_map_class_edgeworth_abs.png'))
#plt.close()

#With the same correlation method for all values (by default, Pearson)
# Identify the correlation coefficient between each attribute and the Class (above we already have the corr_matrix defined using .corr())
l1=[]
for x in dt.columns:
    correlation_x_class = corr_matrix.loc[x, 'Class']
    l1.append(correlation_x_class)
#print(l1)

# Get correlations with the class
corr_with_class = corr_matrix[['Class']]

#Draw a heat map of the correlation matrix (Total)
#Heat Map without Annotations (makes viewing easier)
'''plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlation Matrix - No Annotations')'''
#output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Mapas_calor_corr'
#plt.savefig(os.path.join(output_dir, f'Heat_map_corr_matrix.png'))
#plt.close()

#Draw a heat map of the correlation matrix (Only with Class)
'''plt.figure(figsize=(12, 10))
sns.heatmap(corr_with_class, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlation Matrix - Annotations')'''
#output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Mapas_calor_corr'
#plt.savefig(os.path.join(output_dir, f'Heat_map_class.png'))
#plt.close()

#Make a heat graph with normalized and absolute values

l3=[]
for x in dt.columns:
    correlation_x_class = corr_matrix.loc[x, 'Class']
    l3.append(abs(correlation_x_class))

l3.pop()

l2=[]
for x in l3:
    l2.append((abs(x)*(1/max(l3)))*0.75)
l2.append(1)

# Create a DataFrame with the correlation values
corr_with_class_df = pd.DataFrame(data=l2, index=dt.columns, columns=['Class'])

# Draw the heatmap
'''plt.figure(figsize=(12, 10))
sns.heatmap(corr_with_class_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlation Matrix - Annotations')'''
#output_dir = r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Mapas_calor_corr'
#plt.savefig(os.path.join(output_dir, 'Heat_map_class_norm_abs.png'))
#plt.close()

#it's not a good idea to make the values absolute because we don't know the true relationship with the class

#Filtering out highly correlated features

# Create the correlation matrix and take the absolute value
corr = corr_with_class.abs()

# Create a boolean mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Define the colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
'''sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f", cbar_kws={"shrink": .5})'''
#output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Mapas_calor_corr'
#plt.savefig(os.path.join(output_dir, f'Heat_map_class_abs.png'))
#plt.close()

# Set the correlation coefficient threshold
threshold = 0.1

# Identify columns to drop based on their correlation with 'Class'
to_drop = [column for column in corr_matrix.columns if abs(corr_matrix.loc[column, 'Class']) < threshold]

# Drop the identified columns from the DataFrame
dt.drop(columns=to_drop, inplace=True)

# Optionally, print the columns that are being dropped
#print(f"Columns dropped: {to_drop}")

#Data Preprocessing:

#Calculate the Z-score (only for numeric columns)

#Get only numeric columns that are not in the removed columns list
number_cols3 = [col for col in number_cols2 if col not in to_drop]
#print(len(number_cols3))

#Get only non-numeric columns that are not in the removed columns list
columns_not_numeric_used_now_2 = [col for col in columns_not_numeric_used_now if col not in to_drop]
#print(len(columns_not_numeric_used_now_2))

#Create a dic with the desired numeric columns and their results all in a list
data={}
for x in number_cols3:
    data.update({x:dt[x].tolist()})
#print(data)

#Convert strings of numbers to numbers in values in lists
data_no_strings = {}
for x in data:
    if x!='Age':
        dt[x] = pd.to_numeric(dt[x], errors='coerce')
        data_no_strings.update({x: dt[x].tolist()})
    else:
        data_no_strings.update({x: dt[x].tolist()})
#print(data_no_strings)

# Calculate Z-score for all columns
d_z_score={}
for x in data_no_strings:
    z_scores = (dt[x] - dt[x].mean()) / dt[x].std()
    d_z_score[x + '_zscore'] = list(z_scores)
#print(d_z_score)
#dictionary with the z-score columns, which were not added to the DataFrame, are only in this dic as keys, and their values

# Identify outliers
d_outliers={}
for y in d_z_score:
    c=0
    for z in d_z_score[y]:
        if np.abs(z) > 2:
            c+=1
    d_outliers.update({y:c})
#print(d_outliers) #dic with z-score columns and as values the count of values with abs value > 2

outliers=[]
for w in d_outliers:
    if d_outliers[w]>=(dt.shape[0]//2)+1: #to give 83
        outliers.append(w)
#print(outliers) #Gives empty list, because there are no outliers in this DataFrame, being more than half

outliers=[]
for w in d_outliers:
    if d_outliers[w]>=int(dt.shape[0] * 0.10):
        outliers.append(w)
#print(outliers) #Gives empty list, because there are no outliers in this DataFrame, being 10% of the values

outliers=[]
for w in d_outliers:
    if d_outliers[w]>=int(dt.shape[0] * 0.05):
        outliers.append(w)
#print(outliers)
#print(len(outliers))

#Scatter plot for number columns
'''def plot_z_scores(z_scores_dict):
    plt.figure(figsize=(14, 7))
    for col in z_scores_dict:
        plt.scatter(range(len(z_scores_dict[col])), z_scores_dict[col], label=col)
    plt.axhline(y=2, color='r', linestyle='--', label='Threshold (+2)')
    plt.axhline(y=-2, color='r', linestyle='--', label='Threshold (-2)')
    plt.xlabel('Index')
    plt.ylabel('Z-score')
    plt.title('Z-scores of Columns')
    plt.legend()'''
    #output_dir = r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Grafico_pontos_z-score'
    #plt.savefig(os.path.join(output_dir, f'z_scores_plot.png'))
    #plt.close()

# Plot z-scores using the d_z_score dictionary and save the plot as an image
#plot_z_scores(d_z_score)

# Calculate min-max scaling and plot histograms (Preparation for modeling)
# For all columns (Although the non-numeric 0's and 1's stay the same)

# Start Min-Max Scaler
scaler = MinMaxScaler()

# Adjust and transform data
dt_scaled = scaler.fit_transform(dt)

# Convert back to a DataFrame
dt_scaled = pd.DataFrame(dt_scaled, columns=dt.columns)

#print("DataFrame original:")
#print(dt)

#print("\nDataFrame after Min-Max Scaling:")
#print(dt_scaled)

# Plot histogram for each column scaled by Min-Max
'''for column in dt_scaled.columns:
    plt.figure()
    dt_scaled[column].hist()
    plt.title(f'Histogram {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')'''
    #output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Histograma_min_max'
    #plt.savefig(os.path.join(output_dir, f'Histogram_min_max_{column}.png'))
    #plt.close()

# Print the DataFrame to see the results
#print(dt) (Little can be analyzed)

#Machine Learning Model Selection and Model Training

# Feature and target variable
Y = dt['Class']
l2=[]
for x in dt.columns:
    if x!='Class':
        l2.append(x)
X=dt[l2]

#Decision Tree

#Apply the train_test_split to create a train and a test dataset
from sklearn.model_selection import train_test_split

test_size = 0.3
SEED = 123
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=SEED, stratify=Y)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train.tolist())

#Calculate the distribution of the output label (target) in the test and train sets
y_train.value_counts()
y_test.value_counts()

#Create a decision tree model and test its accuracy in the train and the test set
from sklearn.tree import DecisionTreeClassifier

# Create the decision tree model with max_leaf_nodes = 3 and criterion="gini" and random_state = 0
dt_model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion="gini")

#Use the fit() function from the created DT
# This allows to train the model
dt_model.fit(X_train, y_train_encoded)

#Prediction on the training set
#Use the predict() function; apply it to the X_train dataset
dt_preds_train = dt_model.predict(X_train)

#Calculate the percentage of cases correctly classified
#Compare the resulting vector (predicted) with the vector of actual values (y_train)
# Sum all the positive cases and divide by the total number of cases (length of y_train)
# This will yield the accuracy
dt_acc_train = sum(dt_preds_train == y_train) / len(y_train)

#Repeat the previous step for the test dataset
#Prediction of the test set
dt_preds_test = dt_model.predict(X_test)
#Calculate the percentage of cases correctly classified
dt_acc_test = sum(dt_preds_test == y_test) / len(y_test)

#Plot the train and test accuracy
print("Decision Tree Train Accuracy Gini: %.3f" % dt_acc_train)
print("Decision Tree Test Accuracy Gini: %.3f" % dt_acc_test)

#Random Forest

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Create the model
rf_model = RandomForestClassifier(random_state=0)

# Train the model
rf_model.fit(X_train, y_train_encoded)

# Predict on training set
rf_preds_train = rf_model.predict(X_train)

# Calculate accuracy on the training set
rf_acc_train = sum(rf_preds_train == y_train) / len(y_train)

# Predict on test set
rf_preds_test = rf_model.predict(X_test)

# Calculate accuracy in test set
rf_acc_test = sum(rf_preds_test == y_test) / len(y_test)

# Print the accuracies
print("Random Forest Train Accuracy: {:.3f}".format(rf_acc_train))
print("Random Forest Test Accuracy: {:.3f}".format(rf_acc_test))

# SVM

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Adjust and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train the SVM model
svm_model = SVC(probability=True, random_state=0)
svm_model.fit(X_train_scaled, y_train_encoded)

# Make predictions on the training set
svm_preds_train = svm_model.predict(X_train_scaled)
svm_acc_train = accuracy_score(y_train_encoded, svm_preds_train)

# Make predictions on the test set
svm_preds_test = svm_model.predict(X_test_scaled)
svm_acc_test = accuracy_score(y_test_encoded, svm_preds_test)

# Print the accuracies
print("SVM Train Accuracy: {:.3f}".format(svm_acc_train))
print("SVM Test Accuracy: {:.3f}".format(svm_acc_test))

#XGBoost

# Import the necessary libraries
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Create an encoder to transform categorical columns into dummy variables
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit the encoder to the training data and transform the categorical columns
X_train_encoded = encoder.fit_transform(X_train)

# Transform test data using the same encoder
X_test_encoded = encoder.transform(X_test)

# Create the model XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

# Train the XGBoost model on encoded data
xgb_model.fit(X_train_encoded, y_train_encoded)

# Predict on training set
xgb_preds_train = xgb_model.predict(X_train_encoded)

# Calculate accuracy on the training set
xgb_acc_train = sum(xgb_preds_train == y_train_encoded) / len(y_train_encoded)

# Predict on test set
xgb_preds_test = xgb_model.predict(X_test_encoded)

# Calculate accuracy in test set
xgb_acc_test = sum(xgb_preds_test == y_test) / len(y_test)

# Print the accuracies
print("XGBoost Train Accuracy: {:.3f}".format(xgb_acc_train))
print("XGBoost Test Accuracy: {:.3f}".format(xgb_acc_test))

#Logistic Regression

from sklearn.linear_model import LogisticRegression

# Create the logistic regression model with the configuration for a maximum of 100000 iterations
lr_model = LogisticRegression(random_state=0, max_iter=100000)

# Train the model using the scaled training data and encoded classes
lr_model.fit(X_train_scaled, y_train_encoded)

# Predict on training set
lr_preds_train = lr_model.predict(X_train_scaled)

# Calculate accuracy on the training set
lr_acc_train = sum(lr_preds_train == y_train) / len(y_train)

# Predict on the test set using the scaled data
lr_preds_test = lr_model.predict(X_test_scaled)

# Calculate accuracy in test set
lr_acc_test = sum(lr_preds_test == y_test) / len(y_test)

# Print the accuracies
print("Logistic Regression Train Accuracy: {:.3f}".format(lr_acc_train))
print("Logistic Regression Test Accuracy: {:.3f}".format(lr_acc_test))

#Neural Network

from sklearn.neural_network import MLPClassifier

# Convert class labels to integers using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(Y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Neural Network model
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluate accuracy
train_acc = mlp.score(X_train_scaled, y_train)
test_acc = mlp.score(X_test_scaled, y_test)

print(f'Train Accuracy Rede Neural: {train_acc}')
print(f'Test Accuracy Rede Neural: {test_acc}')

#KNN

from sklearn.neighbors import KNeighborsClassifier

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a KNN model
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Evaluate accuracy
train_acc = knn.score(X_train_scaled, y_train)
test_acc = knn.score(X_test_scaled, y_test)

print(f'Train Accuracy KNN: {train_acc}')
print(f'Test Accuracy KNN: {test_acc}')

#Data Evaluation

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Sample data for demonstration
model_names = ["Decision Tree (Gini)", "Random Forest", "SVM", "XGBoost", "Logistic Regression", "Neural Network",
               "KNN"]
train_accuracies = [0.730, 1.000, 0.917, 0.985, 0.826, 1.000, 0.800]
test_accuracies = [0.680, 0.760, 0.697, 0.667, 0.636, 0.660, 0.640]

# Create a DataFrame for accuracies
accuracy_df = pd.DataFrame({
    'Model': model_names,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies
})

# Plotting Train/Test Accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='value', hue='variable', data=pd.melt(accuracy_df, ['Model']))
plt.title('Train and Test Accuracies of Models')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Data Evaluation'
plt.savefig(os.path.join(output_dir, f'Plotting Train_Test Accuracies.png'))
plt.close()

# Sample confusion matrices, ROC curves, precision-recall curves generation
# Assuming y_test and y_pred are already defined

# For demonstration, create random predictions and ground truth
# Replace this with actual predictions and ground truth
y_test = np.random.randint(0, 2, 100)
y_preds = {
    "Decision Tree (Gini)": np.random.randint(0, 2, 100),
    "Random Forest": np.random.randint(0, 2, 100),
    "SVM": np.random.randint(0, 2, 100),
    "XGBoost": np.random.randint(0, 2, 100),
    "Logistic Regression": np.random.randint(0, 2, 100),
    "Neural Network": np.random.randint(0, 2, 100),
    "KNN": np.random.randint(0, 2, 100)
}

# Confusion Matrices
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for idx, model in enumerate(model_names):
    cm = confusion_matrix(y_test, y_preds[model])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
    axes[idx].set_title(f'Confusion Matrix: {model}')
plt.tight_layout()
output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Data Evaluation'
plt.savefig(os.path.join(output_dir, f'Confusion Matrices.png'))
plt.close()

# ROC and Precision-Recall Curves
fig, axes = plt.subplots(2, 1, figsize=(10, 15))

for model in model_names:
    fpr, tpr, _ = roc_curve(y_test, y_preds[model])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

    precision, recall, _ = precision_recall_curve(y_test, y_preds[model])
    axes[1].plot(recall, precision, label=model)

axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_title('ROC Curves')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

axes[1].set_title('Precision-Recall Curves')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend()

plt.tight_layout()
output_dir=r'C:\Users\ineso\Dropbox\EIACD\Trabalho 2\Trabalho2_EIACD\Data Evaluation'
plt.savefig(os.path.join(output_dir, f'ROC and Precision-Recall Curves.png'))
plt.close()