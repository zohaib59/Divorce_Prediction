import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import shap
import joblib

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("divorce_prediction.csv",encoding='Latin1')

data.head()

pd.set_option('display.max_columns', None)

data.head()


##Check for duplicate values
data[data.duplicated()]

#remove duplicates permanent
data.drop_duplicates(inplace=True)

##Check for missing values
data.isnull().sum()

data.drop(columns=['Social_Support_Level'],axis=1,inplace=True)
data.columns

data.info()
## To show Outliers in the data set run the code 

num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


##To check the data distribution 

def class_distribution(data, column_name='Divorce'):
    # Display total counts and percentage for each class
    distribution = data[column_name].value_counts()
    percentage = data[column_name].value_counts(normalize=True) * 100
    
    print(f"Class distribution for '{column_name}':")
    print(distribution)
    print("\nPercentage distribution:")
    print(percentage)

# Call the function to display the distribution for the 'Resigned' column
class_distribution(data, 'Divorce')

#To convert categorical into numerics run this code

from sklearn import preprocessing
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data[col].unique())
    data[col] = label_encoder.transform(data[col])
    print(f'{col} : {data[col].unique()}')


#Segregrating dataset into X and y

X = data.drop("Divorce", axis = 1)

y = data["Divorce"]

X.head()

y.head()


#Splitting the dataset into testing and training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#Standard Scaller
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# ✅ Models dictionary
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Ridge Classifier': RidgeClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(verbosity=0, n_jobs=-1),
    'CatBoost': CatBoostClassifier(verbose=0),
    'LGBM': LGBMClassifier(verbose=-1),
    'Neural Network (MLP)': MLPClassifier(max_iter=500)
}

# ✅ Evaluation function
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\n--- {name} ---")
    print("Training:")
    print(f"  Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"  Recall   : {recall_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"  F1 Score : {f1_score(y_train, y_train_pred, average='weighted'):.4f}")
    print("Testing:")
    print(f"  Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    return accuracy_score(y_test, y_test_pred)

# ✅ Run evaluation for all models
results = {}
for name, model in models.items():
    scaled = "MLP" in name or "Logistic" in name
    acc = evaluate_model(name, model,
                         X_train_scaled if scaled else X_train,
                         y_train,
                         X_test_scaled if scaled else X_test,
                         y_test)
    results[name] = acc
  
#XGBOOST hyper parameter tune

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Your data: X, y
# Split again if needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define base model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter space
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

# Random search with 5-fold CV
random_search = RandomizedSearchCV(
    xgb, 
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
print("Best Hyperparameters:", random_search.best_params_)

# Evaluation
print("\n--- Evaluation on Train ---")
y_train_pred = best_model.predict(X_train)
print(classification_report(y_train, y_train_pred))

print("\n--- Evaluation on Test ---")
y_test_pred = best_model.predict(X_test)
print(classification_report(y_test, y_test_pred))

