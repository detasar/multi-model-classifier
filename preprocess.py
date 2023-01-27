import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(filepath):
    # read in the input CSV file
    df = pd.read_csv(filepath)
    
    # separate column values into numbers and strings
    numerical_columns = df.select_dtypes(include='number').columns
    categorical_columns = df.select_dtypes(exclude='number').columns
    
    # apply z-scores or min-max scaling to numerical columns based on correlation
    for col in numerical_columns:
        if abs(df[col].corr(df['target'])) > 0.3:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
        else:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
    
    # label-encode categorical columns
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
        
    return df
