import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(df):
    # separate target column
    X = df.drop(columns='target')
    y = df['target']
    
    # perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # fit multiple models with different hyperparameters
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_acc = logreg.score(X_test, y_test)
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_acc = dt.score(X_test, y_test)
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    
    # write results to a dataframe
    results = {'Logistic Regression': logreg_acc, 'Decision Tree': dt_acc, 'Random Forest': rf_acc}
