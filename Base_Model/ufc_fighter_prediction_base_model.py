# -*- coding: utf-8 -*-
"""UFC_Fighter_prediction_base_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rIQycrJ6otlVBdSpTZugkO9jC8ICfds-

#**UFC Fight Outcome Prediction Using Random Forest**

##**Step 1: Install and Import Libraries**
"""

# Install necessary libraries (if not already installed)
!pip install pandas numpy scikit-learn seaborn matplotlib

# Import data manipulation libraries
import pandas as pd
import numpy as np

# Import visualization libraries
from IPython.display import display,  HTML
import seaborn as sns
import matplotlib.pyplot as plt

# Import machine learning libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import the Random Forest model
from sklearn.ensemble import RandomForestClassifier

"""##**Step 2: Load and Prepare Data**"""

# Load from local directory if uploaded to Colab
original_df = pd.read_csv('ufc-master.csv')

# Preview the data
original_df.head()

"""##**Step 3: Data Cleaning and preparing**"""

# Load the original dataset again
df = pd.read_csv('ufc-master.csv')

"""###**3.1 Handle Missing Values**"""

# Calculate the percentage of missing values
missing_percentages = df.isnull().mean() * 100

# Create a DataFrame to display the results
missing_table = pd.DataFrame({
    'Column': missing_percentages.index,
    'Missing Percentage': missing_percentages.values
})

# Sort the table by missing percentage in descending order
missing_table = missing_table.sort_values(by='Missing Percentage', ascending=False)

# Display the table nicely in Jupyter Notebook
display(missing_table)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute numerical columns with mean
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Impute categorical columns with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify Missing Values Are Handled
missing_values = df.isnull().sum()

# Convert to a DataFrame for better display
missing_values_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

# Display the missing values as a table
display(HTML("<h3>Missing values after handling:</h3>"))
display(missing_values_df)

"""###**3.2 Encode Categorical Variables**"""

# Define categorical features to encode
categorical_features = ['RedFighter', 'BlueFighter', 'Location', 'Country','WeightClass', 'TitleBout', 'Gender',  'Winner',  'BlueStance',  'RedStance', 'BetterRank',  'Finish', 'FinishDetails', 'FinishRoundTime']

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Preview encode data
df.head()

"""###**3.3 Define Features and Targets**"""

# Define features (exclude target and unnecessary columns)
X = df.drop(['Date', 'Winner', 'Finish', 'FinishRound'], axis=1)

# Define targets
y_winner = df['Winner']
y_method = df['Finish']
y_round = pd.to_numeric(df['FinishRound'], errors='coerce').fillna(0).astype(int)

X.head()

"""###**3.4 Map Data**"""

x_winner = original_df["Winner"]

# Create a mapping dictionary from x_winner (text) to y_winner (numeric)
mapping_dict = dict(zip(x_winner.unique(), y_winner.unique()))

# Convert the mapping dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=["Winner (Text)", "Winner (Numeric)"])

# Display the mapping as a table
display(HTML("<h3>Mapping between 'Winner' text and numeric labels:</h3>"))
display(mapping_df)

x_method = original_df['Finish']

# Create a mapping dictionary from x_method (text) to y_method (numeric)
mapping_dict = dict(zip(x_method.unique(), y_method.unique()))

# Convert the mapping dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=["Finish (Text)", "Finish (Numeric)"])

# Display the mapping as a table
display(HTML("<h3>Mapping between 'Finish' text and numeric labels:</h3>"))
display(mapping_df)

x_round = original_df['FinishRound']

# Create a mapping dictionary from x_round to y_round
mapping_dict = dict(zip(x_round.unique(), y_round.unique()))

# Convert the mapping dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=["Finish Round (Original)", "Finish Round (Mapped)"])

# Display the mapping as a table
display(HTML("<h3>Mapping between 'FinishRound' text and numeric labels:</h3>"))
display(mapping_df)

"""##**Step 4: Split Data Based on Date**

###**4.1 Define Cut-off Dates**
"""

# Define the cut-off date for training and testing
train_end_date = pd.to_datetime('2024-03-31')
test_start_date = pd.to_datetime('2024-04-01')

"""###**4.2 Split the Data**"""

df['Date'] = pd.to_datetime(df['Date'])

# Create training data: fights up to March 31, 2024
df_train = df[df['Date'] <= train_end_date]

# Create testing data: fights from April 1, 2024 onwards
df_test = df[df['Date'] >= test_start_date]

# Create a summary DataFrame
data_summary = pd.DataFrame({
    'Dataset': ['Training', 'Testing'],
    'Number of Records': [len(df_train), len(df_test)]
})

# Display the summary as a table
display(data_summary)

"""###**4.3 Prepare Features and Targets for Training and Testing**"""

# Features and targets for training data
X_train = df_train.drop(['Date', 'Winner', 'Finish', 'FinishRound'], axis=1)
y_train_winner = df_train['Winner']
y_train_method = df_train['Finish']
y_train_round = pd.to_numeric(df_train['FinishRound'], errors='coerce').fillna(0).astype(int)

# Features and targets for testing data
X_test = df_test.drop(['Date', 'Winner', 'Finish', 'FinishRound'], axis=1)
y_test_winner = df_test['Winner']
y_test_method = df_test['Finish']
y_test_round = pd.to_numeric(df_test['FinishRound'], errors='coerce').fillna(0).astype(int)

# Display the X training data
display(HTML("<h3>X Training Data:</h3>"))
X_train.head()

# Display the X test data
display(HTML("<h3>X Test Data:</h3>"))
X_test.head()

"""###**4.4 Scale Features**"""

# Initialize scaler
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(X_train)

# Scale the features using the previously fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled training data back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Display the scaled training DataFrames
display(HTML("<h3>Scaled Training Data:</h3>"))
display(X_train_scaled_df)

# Convert scaled test data back to DataFrames
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Display the scaled test DataFrames
display(HTML("<h3>Scaled Test Data:</h3>"))
display(X_test_scaled_df)

"""##**Step 5: Model Training**

###**5.1 Winner Prediction Model**
"""

# Initialize Random Forest Classifier for winner prediction
winner_model = RandomForestClassifier(random_state=42)

# Train the model
winner_model.fit(X_train_scaled, y_train_winner)

"""###**5.2 Method Prediction Model**"""

# Initialize Random Forest Classifier for method prediction
method_model = RandomForestClassifier(random_state=42)

# Train the model
method_model.fit(X_train_scaled, y_train_method)

"""###**5.3 Round Prediction Model**"""

# Initialize Random Forest Classifier for round prediction
round_model = RandomForestClassifier(random_state=42)

# Train the model
round_model.fit(X_train_scaled, y_train_round)

"""##**Step 6: Make Predictions**

###**6.1 Winner Predictions**
"""

# Predict winner labels
y_pred_winner = winner_model.predict(X_test_scaled)

# Predict winner probabilities
y_proba_winner = winner_model.predict_proba(X_test_scaled)
winner_probs = y_proba_winner.max(axis=1)

"""###**6.2 Method Predictions**"""

# Predict method labels
y_pred_method = method_model.predict(X_test_scaled)

# Predict method probabilities
y_proba_method = method_model.predict_proba(X_test_scaled)
method_probs = y_proba_method.max(axis=1)

"""###**6.3 Round Predictions**"""

# Predict round labels
y_pred_round = round_model.predict(X_test_scaled)

# Predict round probabilities
y_proba_round = round_model.predict_proba(X_test_scaled)
round_probs = y_proba_round.max(axis=1)

"""##**Step 7: Evaluate the Models**

###**7.1 Winner Model Evaluations**
"""

# Accuracy
accuracy_winner = accuracy_score(y_test_winner, y_pred_winner)
#print(f"Winner Model Accuracy: {accuracy_winner:.2f}")
display(HTML(f"<h3>Winner Model Accuracy: {accuracy_winner:.2f}</h3>"))

# Convert classification report to DataFrame
report = classification_report(y_test_winner, y_pred_winner, output_dict=True)
classification_df = pd.DataFrame(report).transpose()

# Display Classification Report as a Table
#print("Winner Model Classification Report:")
display(HTML("<h3>Winner Model Classification Report:</h3>"))
display(classification_df)

# Confusion Matrix as a Heatmap
# Blue = 0 and Red =1
cm_winner = confusion_matrix(y_test_winner, y_pred_winner)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_winner, annot=True, fmt='d', cmap='Blues', xticklabels=["Blue", "Red"], yticklabels=["Blue", "Red"])
plt.title("Winner Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""###**7.2 Method Model Evaluation**"""

# Accuracy
accuracy_method = accuracy_score(y_test_method, y_pred_method)
display(HTML(f"<h3>Method Model Accuracy: {accuracy_method:.2f}</h3>"))

# Convert classification report to DataFrame
report_method = classification_report(y_test_method, y_pred_method, output_dict=True)
classification_method_df = pd.DataFrame(report_method).transpose()

# Display Classification Report as a Table
display(HTML("<h3>Method Model Classification Report:</h3>"))
display(classification_method_df)

# Confusion Matrix as a Heatmap
cm_method = confusion_matrix(y_test_method, y_pred_method)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_method, annot=True, fmt='d', cmap='Blues',
            xticklabels=["DQ", "KO/TKO",  "M-DEC", "NaN", "S-DEC", "SUB", "U-DEC"],
            yticklabels=["DQ", "KO/TKO",  "M-DEC", "NaN", "S-DEC", "SUB", "U-DEC"])
plt.title("Method Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""###**7.3 Round Model Evaluation**"""

# Accuracy
accuracy_round = accuracy_score(y_test_round, y_pred_round)
display(HTML(f"<h3>Round Model Accuracy: {accuracy_round:.2f}</h3>"))

# Convert classification report to DataFrame
report_round = classification_report(y_test_round, y_pred_round, output_dict=True)
classification_round_df = pd.DataFrame(report_round).transpose()

# Display Classification Report as a Table
display(HTML("<h3>Round Model Classification Report:</h3>"))
display(classification_round_df)

# Confusion Matrix as a Heatmap
cm_round = confusion_matrix(y_test_round, y_pred_round)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_round, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Round 1", "Round 2", "Round 3", "Round 4", "Round 5"],
            yticklabels=["Round 1", "Round 2", "Round 3", "Round 4", "Round 5"])
plt.title("Round Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()