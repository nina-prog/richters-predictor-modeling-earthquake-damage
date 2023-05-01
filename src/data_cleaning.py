# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn import preprocessing


def get_pearson_correlated_features(data=None, threshold=0.7):
    """
    Calculates the pearson correlation of all features in the dataframe and returns a set of features with a correlation greater than the threshold. 
    
    :param data: Dataframe with features and values
    :param threshold: A number between 0 and 1
    
    :returns A set of correlated feature names
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    
    return correlated_features


def get_cramers_v_correlated_features(data=None, threshold=0.7):
    """
    Calculates the cramers V correlation of all features and returns a set of features with a correlation greater than the threshold. 
    Cramers V is based on Chi square, for reference see: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Note that this function is desined to work for categorical features only!
    Code was copied and modified from this source: https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix/notebook
    
    :param data: Dataframe with features and values
    :param threshold: A number between 0 and 1
    
    :returns A set of correlated feature names
    """
    # Encode features
    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame() 

    for i in data.columns :
        data_encoded[i]=label.fit_transform(data[i])

    # Internal function to calculate cramers V for two features
    def _cramers_V(var1, var2) :
        crosstab = np.array(pd.crosstab(var1,var2, rownames=None, colnames=None))  # Cross table building
        stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
        obs = np.sum(crosstab)  # Number of observations
        mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
        return (stat / (obs * mini))
        #return stat

    # Calculate values for each pair of features
    rows= []
    for var1 in data_encoded:
        col = []
        for var2 in data_encoded :
            cramers = _cramers_V(data_encoded[var1], data_encoded[var2])  # Cramer's V test
            col.append(round(cramers, 4))  # Keeping of the rounded value of the Cramer's V  
        rows.append(col)
    
    # Create a pandas df from the results
    cramers_results = np.array(rows)
    corr_matrix = pd.DataFrame(cramers_results, columns = data_encoded.columns, index =data_encoded.columns)
    
    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    
    return correlated_features


def get_mcc_correlated_features(data=None, threshold=0.7):
    """
    Calculates the MCC correlation of all features and returns a set of features with a correlation greater than the threshold. 
    Cramers V is based on Chi square, for reference see: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Note that this function is desined to work for categorical features only!
    Code was copied and modified from this source: https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix/notebook
    
    :param data: Dataframe with features and values
    :param threshold: A number between 0 and 1
    
    :returns A set of correlated feature names
    """
    # Encode features
    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame() 

    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame() 
    
    for c in data.columns:
        if c in data.columns:
            data_encoded[c] = label.fit_transform(data[c])
        else:
            data_encoded[c] = data[c]
    
    # Calculate values for each pair of features
    rows= []
    for var1 in data_encoded:
        col = []
        for var2 in data_encoded :
            phi = matthews_corrcoef(data_encoded[var1], data_encoded[var2])  
            col.append(phi)  # phi  
        rows.append(col)
    
    # Create a pandas df from the results
    phi_results = np.array(rows)
    corr_matrix = pd.DataFrame(phi_results, columns=data_encoded.columns, index=data_encoded.columns)
    
    # Get the set of correlated features
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    
    return correlated_features


def drop_correlated_features(data=None, config=None):
    """
    Gets the correlated features according to the configuration and drops them from the provided dataframe. 
    Then the dataframe without the correlated features is returned. 
    
    Example for the config: 
    [
        {
            'feature_names': <List of feature names>,
            'threshold'    : A number between 0 and 1
            'method'       : <One out of the set {'MCC', 'CramesV', 'Pearson'}>
        },
        {
            'feature_names': <List of feature names>,
            'threshold'    : A number between 0 and 1
            'method'       : <One out of the set {'MCC', 'CramesV', 'Pearson'}>
        }, ...
    ]
    
    :param data: The dataframe to drop the features from
    :param config: A list of dicts. Every dict has to contain the keys 'feature_names', 'method' and 'threshold'. 
                   The 'feature_names' determine of which features the correlation is calculated. 
                   Method has to be one out of the set {'MCC', 'CramesV', 'Pearson'}.
                   The value of method determines the function which is used to calculate the correlations.
                   Only features with a higher correlation than 'threshold' will be dropped. 
                   
    :returns A dataframe without the correlated features
    """
    # Traverse all dicts in the config
    # Note: This could be parallelized
    for d in config:
        if d['method'] == 'MCC':
            features_to_drop = get_mcc_correlated_features(data=data[d['feature_names']], threshold=d['threshold'])
        elif d['method'] == 'CramersV':
            features_to_drop = get_cramers_v_correlated_features(data=data[d['feature_names']], threshold=d['threshold'])
        elif d['method'] == 'Pearson':
            features_to_drop = get_pearson_correlated_features(data=data[d['feature_names']], threshold=d['threshold'])
        else: 
            print(f"Correlation method '{d['method']}' is not implemented.")
        
        # Drop features
        if len(features_to_drop) > 0:
            data = data.drop(features_to_drop, axis=1)
    
    return data

def format_dtypes(df):
    """
    Formats the data types of columns in a pandas DataFrame.

    :param df: pandas DataFrame to be processed
    :type df: pandas.DataFrame

    :return: pandas DataFrame with formatted data types
    :rtype: pandas.DataFrame
    """
    # categorical values
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df[cat_cols] = df[cat_cols].astype('category')

    return df

def handle_outliers_IQR(df, ignore_cols, iqr_factor=1.5, method="soft_drop"):
    """
    Handle outliers in a mixed categorical and numerical dataframe using the IQR method. Note that this function only drops outliers based on IQR and does not consider any categorical features. It is assumed that any categorical features in the dataframe will not contribute to the detection of outliers.

    :param df: The input dataframe.
    :type df: pd.DataFrame
    :param iqr_factor: The factor to be multiplied with the IQR to determine the outlier bounds.
    :type iqr_factor: float, optional (default=1.5)
    :param method: The method to handle outliers. The options are:
        - "drop": drops the rows with outliers.
        - "cap": replaces the outliers with the upper or lower limit.
    :type method: str, optional (default="drop")
    :return: The processed dataframe with outliers handled.
    :rtype: pd.DataFrame
    """
    # Identify the numerical columns in the dataframe
    df = df.drop(ignore_cols, axis=1)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Calculate the IQR for each numerical column
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR

    # Identify the rows where any numerical value is outside the bounds aka identify the outliers
    outlier_mask = (df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)
    outlier_rows = outlier_mask.any(axis=1)

    if method == "hard_drop":
        # Drop rows with outliers
        df = df[~outlier_rows]
    elif method == "soft_drop":
        # Compute the number of outliers per row and drop rows based on the threshold (30%)
        count_outliers = outlier_mask.notna().sum(axis=1)
        drop_indices = count_outliers[count_outliers >= 30 * len(numeric_cols)].index
        df = df.drop(drop_indices)
    elif method == "replace":
        upper_limit = df[numeric_cols].mean() + 3 * df[numeric_cols].std()
        lower_limit = df[numeric_cols].mean() - 3 * df[numeric_cols].std()
        # Replace outliers with upper or lower limit
        df[numeric_cols] = np.where(df[numeric_cols] > upper_limit, upper_limit,
                                    np.where(df[numeric_cols] < lower_limit, lower_limit,
                                             df[numeric_cols]))
    else:
        raise ValueError(f"Invalid method: {method}, must be either 'drop' or 'replace'.")

    return df

def clean_data(df, config, ignore_cols, outlier_method):
    """
    Cleans the input pandas DataFrame by dropping unnecessary columns, formatting column data types, and handling outliers.

    :param df: pandas DataFrame to be processed
    :type df: pandas.DataFrame
    :param config: configparser object containing configuration settings
    :type config: configparser.ConfigParser
    :param ignore_cols: list of column names to ignore when handling outliers (default: None)
    :type ignore_cols: list or None
    :param outlier_method: The method to handle outliers. The options are:
        - "drop": drops the rows with outliers.
        - "cap": replaces the outliers with the upper or lower limit.
    :type outlier_method: str
    :return: cleaned pandas DataFrame
    :rtype: pandas.DataFrame
    """
    # Drop unnecessary columns
    drop_cols = config.get("data_cleaning", "NO DATA CLEANING DEFINED").get("columns_to_remove")
    df = df.drop(drop_cols, axis=1)
    # Format column types
    df = format_dtypes(df)
    # Handle outliers
    df = handle_outliers_IQR(df, ignore_cols=ignore_cols, method=outlier_method)

    return df