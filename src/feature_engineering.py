import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def encode_train_data(x_train: pd.DataFrame):
    """
    Encodes the train data using sklearn One Hot Encoded and returns fitted OHE object.
    We need the fitted OHE object to transform our test data thus we also return the OHE object.

    :param x_train:
    :return: Encoded DataFrame and fitted OHE object
    """
    x_train_cats = x_train.select_dtypes(['object', 'category'])

    # Fit One Hot Encoding Object
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.int64)
    x_train_cats_encoded = ohe.fit_transform(x_train_cats).toarray()

    # Transform encoded data to pandas dataframe
    x_train_cats_encoded = pd.DataFrame(x_train_cats_encoded, columns=ohe.get_feature_names_out(), index=x_train.index)

    # Drop old features
    feats_to_drop = list(ohe.feature_names_in_)
    x_train = x_train.drop(columns=feats_to_drop, axis=1)

    # Concat old dataframe with new encoded features
    x_train_encoded = pd.concat([x_train, x_train_cats_encoded], axis=1)

    return x_train_encoded, ohe


def encode_test_data(x_test: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """
    Applies the already fitted OHE object on the test dataframe x_test.
    First extracts categorical columns from the x_test and transforms them using the ohe object.
    Then the encoded data gets concatenated with the remaining not encoded features.

    :param x_test: Test DataFrame to transform using the fitted OHE object
    :param ohe: Fitted OneHotEncoder Object yielded from 'encode_train_data()' function
    :return: Encoded DataFrame
    """
    # Get categorical columns and transform them using already fitted OHE object
    x_test_cats = x_test.select_dtypes(['object', 'category'])
    x_test_cats_encoded = ohe.transform(x_test_cats).toarray()

    # Transform to pandas DataFrame
    x_test_cats_encoded = pd.DataFrame(x_test_cats_encoded, columns=ohe.get_feature_names_out(), index=x_test.index)

    # Drop old features
    feats_to_drop = list(ohe.feature_names_in_)
    x_test = x_test.drop(columns=feats_to_drop, axis=1)

    # Concat old dataframe with new encoded features
    x_test_encoded = pd.concat([x_test, x_test_cats_encoded], axis=1)

    return x_test_encoded


def normalize_train_data(x_train: pd.DataFrame, method: str = "standard"):
    """
    Function to normalize the train data.
    Fits StandardScaler on given train DataFrame and also outputs the scaler.

    :param x_train: train DataFrame
    :param method: Method to scale, either 'standard' or 'minmax'
    :return: Scaled Train DataFrame
    """
    assert method in ["standard", "minmax"], print("method must either be 'standard' or 'minmax'")

    scaler = None
    if method=="standard":
        scaler = StandardScaler()
    if method=="minmax":
        scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    # Transform back to pandas DataFrame
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    return x_train_scaled, scaler


def normalize_test_data(x_test: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Function to normalize the test data. Uses already fitted StandardScaler or MinMax scaler object
    to transform given DataFrame.

    :param x_test: test DataFrame
    :param scaler: Fitted StandardScaler or MinMax Scaler object, yielded from 'normalize_train_data' function
    :return: Scaled Test DataFrame
    """

    x_test_scaled = scaler.transform(x_test)
    # Transform back to pandas DataFrame
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    return x_test_scaled