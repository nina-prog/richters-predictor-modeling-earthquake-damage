import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif


def get_top_k_features_using_rfe(x_train: pd.DataFrame, y_train: pd.DataFrame, k = 0.50, step: int = 2, verbose: int = 0):
    """
    Applies Recursive Feature Elimination to get the best k features. Note that k can either be a integer, or a float.
    If it is a float, e.g. 0.5, then 50% of the total features, that are estimated to be relevant, will be selected.
    Uses a RandomForest as a classifier and therefore the _feature_importances as a scoring.

    :param x_train: DataFrame of the training data
    :param y_train: Dataframe of the labels
    :param k: Int or float: Number of features to be selected. The smaller it is, the more computation power is needed.
    :param step: Step size of the feature elimination. The smaller it is, the more computation power is needed.
    :param verbose: Verbosity level. Set to 0 to be quiet.

    :return: List of k top features, Fitted RFE object
    """

    # Define classifier
    clf = RandomForestClassifier()
    x_train = pd.get_dummies(x_train)
    rfe = RFE(estimator=clf, n_features_to_select=k, step=step, verbose=verbose)
    rfe.fit(x_train, y_train["damage_grade"].values.flatten())

    # Select feature name that are estimated to be the best features
    ranking = pd.DataFrame({"feature": x_train.columns, "rfe_support": rfe.support_})
    best_k_features = ranking[ranking["rfe_support"] == True]["feature"].to_list()

    return best_k_features, rfe


def plot_rfe_ranking(rfe: RFE):
    """
    Plots results from RFE, i.e. feature and its rank in barplot.

    :param rfe: Fitted RFE object; Gets returned from 'get_top_k_features_using_rfe' function
    :return: None
    """

    # Create dataframe for plotting
    res = pd.DataFrame({"feature": rfe.feature_names_in_, "rfe_rank": rfe.ranking_})
    res = res.sort_values(by="rfe_rank")

    # Plot settings
    plt.figure(figsize=(20, 8), dpi=128)
    plt.title(f"Rank of Features by RFE (n_features = {rfe.n_features_} from {rfe.n_features_in_} in Total)", size=24)
    g = sns.barplot(data=res, x="feature", y="rfe_rank", edgecolor="black", palette="Spectral")
    plt.xticks(rotation=90)
    plt.bar_label(g.containers[0], padding=1.5)
    plt.show()


def get_top_k_features_using_mi(x_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 30):
    """
    Computes the Mutual Information to the label and outputs the top k features as a list

    :param x_train: DataFrame of the train values
    :param y_train: DataFrame of the labels
    :param k: Number of top k features to output as list

    :return: List of top k features based on Mutual Information score
    """

    # Compute mutual information scores
    mi_scores = mutual_info_classif(x_train, y_train["damage_grade"].values.flatten())
    mi_scores = pd.Series(mi_scores)
    mi_scores.index = x_train.columns

    # Sorts descending based on mi score and gets k first features
    top_k_features =  mi_scores.sort_values(ascending=False)[: k].keys().tolist()

    return top_k_features, mi_scores


def plot_mi_ranking(mi_scores):
    """
    Plots results of the Mutual Information Score.

    :param mi_scores: Pandas Series -- Output of the get_top_k_features_using_mi function
    :return: None
    """

    mi_scores.sort_values(ascending=False, inplace=True)
    plt.figure(figsize=(20,8), dpi=128)
    plt.title("Mutual Information Scores Ordered Descending", size=24)
    g = sns.barplot(x=mi_scores.keys(), y=mi_scores.values, palette="Spectral", edgecolor="black")
    plt.xticks(rotation=90)
    plt.show()