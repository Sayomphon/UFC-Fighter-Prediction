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

