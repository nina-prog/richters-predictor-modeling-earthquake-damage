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