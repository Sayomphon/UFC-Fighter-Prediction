# -*- coding: utf-8 -*-
"""UFC_Fighter_prediction_best_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Okr4OI1NC1S-eU3iOxsX1egGC3E99xWz

#**UFC Fight Outcome Prediction Using multi models**

##**Step 1: Install and Import Libraries**
"""

# Install necessary libraries (if not already installed)
!pip install pandas numpy scikit-learn seaborn matplotlib catboost tensorflow

# Import data manipulation libraries
import pandas as pd
import numpy as np

# Import visualization libraries
from IPython.display import display,  HTML
import seaborn as sns
import matplotlib.pyplot as plt

# Import machine learning libraries
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import  cross_val_score

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Import load model
from tensorflow.keras.models import load_model

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

# Load the selected features CSV file
selected_features = pd.read_csv('selected_features_40.csv')

# Load the selected features
features_to_use = selected_features['Selected Features'].tolist()

# Features and targets for training data
X_train = df_train[features_to_use]
y_train_winner = df_train['Winner']
y_train_method = df_train['Finish']
y_train_round = pd.to_numeric(df_train['FinishRound'], errors='coerce').fillna(0).astype(int)

# Features and targets for testing data
X_test = df_test[features_to_use]
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

"""##**Step 5: Load models training**

###**5.1 Deep Neural network Model**
"""

def train_deep_nn_model(X_train, y_train, input_dim, output_dim, loss_function, activation='relu', final_activation='sigmoid', epochs=50, batch_size=32, class_weights=None):
    model = Sequential()
    model.add(Dense(256, activation=activation, input_shape=(input_dim,)))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=activation))
    model.add(Dense(output_dim, activation=final_activation))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_function, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, class_weight=class_weights)
    return model

"""###**5.2 Machine learning Models**"""

# Define the models to train
model_classes = {
    "Random Forest": lambda: RandomForestClassifier(random_state=42),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "CatBoost": lambda: CatBoostClassifier(verbose=0),
    "XGB": lambda: XGBClassifier(random_state=42)
}

"""##**Step 6: Training Models**

###**6.1 Training Winner deep neural network models**
"""

# Train for y_train_winner
deep_nn_model_winner = train_deep_nn_model(
    X_train_scaled,
    y_train_winner,
    input_dim=X_train_scaled.shape[1],
    output_dim=1,
    loss_function='binary_crossentropy',
    final_activation='sigmoid',
    epochs=50,
    batch_size=32
)

# Save the models later for evaluation or inference
deep_nn_model_winner.save('deep_nn_model_winner.h5')

"""###**6.2 Training Method deep neural network Models**"""

# Train for y_train_method
deep_nn_model_method = train_deep_nn_model(
    X_train_scaled,
    y_train_method,
    input_dim=X_train_scaled.shape[1],
    output_dim=len(np.unique(y_train_method)),
    loss_function='sparse_categorical_crossentropy',
    final_activation='softmax',
    epochs=50,
    batch_size=32
)

# Save the models later for evaluation or inference
deep_nn_model_method.save('deep_nn_model_method.h5')

"""###**6.3 Training multi Models**"""

# Define a function to train models
def train_models(model_classes, X_train, y_train):
    models = {}
    for name, model_class in model_classes.items():
        model = model_class()
        model.fit(X_train, y_train)
        models[name] = model
    return models

# Training 'winner' task (binary classification)
models_winner = train_models(model_classes, X_train_scaled, y_train_winner)

# Training 'method' task (multi-class classification)
models_method = train_models(model_classes, X_train_scaled, y_train_method)

"""#**7. Testing models**

###**7.1 Evaluate trained DL model**
"""

def evaluate_dl_model_on_test(model, X_test, y_test, is_binary_classification=True):
    # Predict probabilities
    y_pred_prob = model.predict(X_test)

    # Convert probabilities to class predictions
    if is_binary_classification:
        y_pred = (y_pred_prob > 0.5).astype(int)  # For binary classification
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)  # For multi-class classification

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    display(HTML(f'<h3>Deep learning model test accuracy: {accuracy:.4f}</h3>'))
    return accuracy

# Test DL model for 'winner' task (binary classification)
display(HTML('<h3>Evaluating Winner deep learning model...</h3>'))
dl_accuracy_winner = evaluate_dl_model_on_test(deep_nn_model_winner, X_test_scaled, y_test_winner, is_binary_classification=True)

# Test DL model for 'method' task (multi-class classification)
display(HTML('<h3>Evaluating Method deep learning model...</h3>'))
dl_accuracy_method = evaluate_dl_model_on_test(deep_nn_model_method, X_test_scaled, y_test_method, is_binary_classification=False)

"""###**7.2 Evaluate trained ML models**"""

# Define a function to evaluate models
def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Test Accuracy": accuracy})
    return pd.DataFrame(results)

# Test ML models for 'winner' task (binary classification)
ml_results_winner = evaluate_models(models_winner, X_test_scaled, y_test_winner)
display(ml_results_winner)

# Test ML models for 'method' task (multi-class classification)
ml_results_method = evaluate_models(models_method, X_test_scaled, y_test_method)
display(ml_results_method)

"""#**8. Compare Winner models accuracy**"""

# Ensure consistency in lengths and construct DataFrame for the winner task
if len(ml_results_winner['Model']) == len(ml_results_winner['Test Accuracy']):
    results_winner = pd.DataFrame({
        'Model': ml_results_winner['Model'].tolist() + ['Deep Learning'],
        'Test Accuracy': ml_results_winner['Test Accuracy'].tolist() + [dl_accuracy_winner]
    })
    # Sort by Test Accuracy in descending order
    results_winner = results_winner.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)
else:
    print("Mismatch in lengths for winner task. Check the input data.")
    results_winner = pd.DataFrame(columns=['Model', 'Test Accuracy'])

# Ensure consistency in lengths and construct DataFrame for the method task
if len(ml_results_method['Model']) == len(ml_results_method['Test Accuracy']):
    results_method = pd.DataFrame({
        'Model': ml_results_method['Model'].tolist() + ['Deep Learning'],
        'Test Accuracy': ml_results_method['Test Accuracy'].tolist() + [dl_accuracy_method]
    })
    # Sort by Test Accuracy in descending order
    results_method = results_method.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)
else:
    print("Mismatch in lengths for method task. Check the input data.")
    results_method = pd.DataFrame(columns=['Model', 'Test Accuracy'])

# Display the results
display(HTML('<h3>Winner Task Results</h3>'))
display(results_winner)

display(HTML('<h3>Method Task Results</h3>'))
display(results_method)