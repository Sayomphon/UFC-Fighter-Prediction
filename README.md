# UFC-Fighter-Prediction

### 1. Data Collection and Preparation
  - Use the following scraper as a starting point: [https://github.com/remypereira99/UFC-Web-Scraping.](https://github.com/remypereira99/UFC-Web-Scraping)
  - Supplement this data with additional sources, such as [Kaggle UFC Datase](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)t or other relevant MMA data.
### 2. Data Analysis and Insights
  - Analyze historical fight data to identify patterns and trends.
  - Incorporate relevant inputs such as:
    - Fighter stats (e.g., height, reach, weight, fighting style, and record).
    - Venue characteristics (e.g., altitude, location).
    - Betting-related inputs (e.g., betting odds, over/under rounds).
    - Referee tendencies (e.g., stricter or lenient officiating style).
### 3. Model Training and Testing
  - Train your model using data up to March 2024.
  - Use fight results from April 2024 to present for testing the model.
### 4. Model Outputs
  - Winner (Probability)
  - Method (Probability)
  - Round (Probability)

# UFC Fight Outcome Prediction Using multi models

## Step 1: Install and Import Libraries

- **Installation:** Start by installing necessary libraries using `pip install`. This ensures all required packages are available.
- **Imports:** Import libraries for data manipulation (`pandas`, `numpy`), visualization (`seaborn`, `matplotlib`), display (`IPython.display`), machine learning (`scikit-learn`), and specific models (`CatBoostClassifier`, `XGBClassifier`, etc.).

``` python
!pip install pandas numpy scikit-learn seaborn matplotlib catboost tensorflow
```

``` python
"""
Purpose: Import all necessary libraries for data manipulation, visualization, preprocessing, and machine learning models.

Variables:
    pd: Alias for pandas, used for data manipulation.
    np: Alias for numpy, used for numerical operations.
    sns: Alias for seaborn, used for data visualization.
    plt: Alias for matplotlib.pyplot, used for plotting graphs.
    Machine learning models and tools are imported from sklearn, catboost, and xgboost.
"""

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
from sklearn.model_selection import  cross_val_score

# Import models
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
```

## Step 2: Load and Prepare Data

- **Loading Data:** You load the dataset `ufc-master.csv` into a pandas DataFrame called `original_df`.
- **Previewing Data:** Display the first few rows using `head()` to understand the structure of the dataset.

``` python
"""
Purpose: Load the UFC dataset from a CSV file into a pandas DataFrame and preview the first few rows to understand the data structure.

Variables:
    original_df: The original DataFrame containing the loaded data.
"""

# Load from local directory if uploaded to Colab
original_df = pd.read_csv('ufc-master.csv')

# Preview the data
original_df.head()
```

## Step 3: Data Cleaning and preparing

``` python
"""
Purpose: Reload the dataset into a new DataFrame for cleaning, keeping the original data intact.

Variables:
    df: DataFrame used for data cleaning and preprocessing.
"""

# Load the original dataset again
df = pd.read_csv('ufc-master.csv')
```

### 3.1 Handle Missing Values
- **Calculating Missing Percentages:** Calculate the percentage of missing values for each column to identify columns that need imputation.
- **Imputation:**
  - **Numerical Columns:** Missing values are filled with the mean of each column.
  - **Categorical Columns:** Missing values are filled with the mode (most frequent value) of each column.
- **Verification:** After imputation, you verify that there are no missing values left.

``` python
"""
Purpose: Calculate and display the percentage of missing values in each column to identify columns that need imputation.

Variables:
    missing_percentages: Series containing the percentage of missing values per column.
    missing_table: DataFrame that tabulates columns and their missing percentages.
"""

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
```

``` python
"""
Purpose:
    - Identify numerical and categorical columns for appropriate imputation.
    - Impute missing numerical values with the mean and categorical values with the mode.

Variables:
    numerical_cols: List of numerical column names.
    categorical_cols: List of categorical column names.
"""

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute numerical columns with mean
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Impute categorical columns with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

``` python
"""
Purpose:
    - Verify that all missing values have been handled.

Variables:
    - missing_values: Series with the count of missing values per column.
    - missing_values_df: DataFrame for displaying missing values after imputation.
"""
missing_values = df.isnull().sum()

# Convert to a DataFrame for better display
missing_values_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

# Display the missing values as a table
display(HTML("<h3>Missing values after handling:</h3>"))
display(missing_values_df)
```

### 3.2 Encode Categorical Variables
- **Label Encoding:** Categorical variables are converted into numerical format using `LabelEncoder`. This is necessary because most machine learning models require numerical input.

``` python
"""
Purpose: Convert categorical variables into numerical format using Label Encoding.

Variables:
    - categorical_features: List of categorical column names to be encoded.
    - le: Instance of LabelEncoder.
    - df[col]: Each categorical column is transformed using Label Encoding.
"""

# Define categorical features to encode
categorical_features = ['RedFighter', 'BlueFighter', 'Location', 'Country','WeightClass', 'TitleBout', 'Gender',  'Winner',  'BlueStance',  'RedStance', 'BetterRank',  'Finish', 'FinishDetails', 'FinishRoundTime']

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Preview encode data
df.head()
```

### 3.3 Define Features and Targets
- **Features (`x`):** Define the feature set by dropping unnecessary columns like `'Date'`, `'Winner'`, `'Finish'`, and `'FinishRound'`.
- **Targets (`y`):** 
  - **`y_winner`:** The target variable for predicting the winner.
  - **`y_method`:** The target variable for predicting the method of victory.
  - **`y_round`:** The target variable for predicting the round in which the fight ends.

``` python
"""
Purpose: Separate the dataset into features (X) and target variables (y_winner, y_method, y_round) for modeling.

Variables:
    - X: Features used for training the models.
    - y_winner: Target variable for predicting the winner.
    - y_method: Target variable for predicting the method of victory.
    - y_round: Target variable for predicting the round in which the fight ends, converted to integer.
"""

# Define features (exclude target and unnecessary columns)
X = df.drop(['Date', 'Winner', 'Finish', 'FinishRound'], axis=1)

# Define targets
y_winner = df['Winner']
y_method = df['Finish']
y_round = pd.to_numeric(df['FinishRound'], errors='coerce').fillna(0).astype(int)
```

### 3.4 Map Data
- **Creating Mappings:** Create mappings between the original text labels and the encoded numerical labels for `'Winner'`, `'Finish'`, and `'FinishRound'`.This is helpful for interpreting the model's predictions later.

``` python
"""
Purpose: Create a mapping between the original text labels and the encoded numerical labels for the 'Winner' column.

Variables:
    - x_winner: Original 'Winner' column with text labels.
    - mapping_dict: Dictionary mapping text labels to numerical labels.
    - mapping_df: DataFrame displaying the mapping.

Repeat similar steps for x_method and x_round to create mappings for 'Finish' and 'FinishRound' columns.
"""

x_winner = original_df["Winner"]

# Create a mapping dictionary from x_winner (text) to y_winner (numeric)
mapping_dict = dict(zip(x_winner.unique(), y_winner.unique()))

# Convert the mapping dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=["Winner (Text)", "Winner (Numeric)"])

# Display the mapping as a table
display(HTML("<h3>Mapping between 'Winner' text and numeric labels:</h3>"))
display(mapping_df)
```

``` python
x_method = original_df['Finish']

# Create a mapping dictionary from x_method (text) to y_method (numeric)
mapping_dict = dict(zip(x_method.unique(), y_method.unique()))

# Convert the mapping dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=["Finish (Text)", "Finish (Numeric)"])

# Display the mapping as a table
display(HTML("<h3>Mapping between 'Finish' text and numeric labels:</h3>"))
display(mapping_df)
```

``` python
x_round = original_df['FinishRound']

# Create a mapping dictionary from x_round to y_round
mapping_dict = dict(zip(x_round.unique(), y_round.unique()))

# Convert the mapping dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=["Finish Round (Original)", "Finish Round (Mapped)"])

# Display the mapping as a table
display(HTML("<h3>Mapping between 'FinishRound' text and numeric labels:</h3>"))
display(mapping_df)
```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

``` python

```

