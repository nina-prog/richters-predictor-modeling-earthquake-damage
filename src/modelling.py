from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import matthews_corrcoef, make_scorer

import numpy as np
import xgboost


def hyperparameter_optimization(model=None, hyperparameter_grid=None, train_data=None, train_labels=None, scoring=None):
    # Get train labels in shape that .fit() expects
    if model == "XGBoost":
        # XGBoost expects [0, 1, 2] instead of [1, 2, 3]
        encoder = LabelEncoder()
        train_labels = encoder.fit_transform(train_labels["damage_grade"].to_numpy())
    else:
        train_labels = train_labels["damage_grade"].ravel()

    # Choose model based on input
    if model == "Dummy":
        model = DummyClassifier(strategy="most_frequent")
    elif model == "RandomForest":
        print("Fitting RandomForest ...")
        model = RandomForestClassifier(random_state=42)
    elif model == "DecisionTree":
        print("Fitting DecisionTree ...")
        model = DecisionTreeClassifier(random_state=42)
    elif model == "XGBoost":
        print("Fitting XGBoost ...")
        #model = xgboost.XGBClassifier(random_state=42, n_jobs=-1)
        model = xgboost.XGBClassifier(n_estimators=100,
                                      max_depth=20,
                                      learning_rate=0.1,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      random_state=42,
                                      n_jobs=-1)

    scoring_string = scoring
    # Only for evaluating the engineered features, change in the week after
    if scoring == "MCC":
        scoring = make_scorer(matthews_corrcoef)
    elif scoring == "ACC":
        scoring = "accuracy"

    cv_results = cross_validate(model, train_data, train_labels, cv=5,
                                scoring=scoring,
                                n_jobs=-1,
                                return_train_score=True)
    model.fit(train_data, train_labels)

    print("")
    print(f"CV Training: {round(np.mean(cv_results['train_score']), 4)} +/- {round(np.std(cv_results['train_score']), 4)} {scoring_string}")
    print(f"CV Test: {round(np.mean(cv_results['test_score']), 4)} +/- {round(np.std(cv_results['test_score']), 4)} {scoring_string}")
    print("")
    
    return model, cv_results


def make_prediction(model=None, test_data=None, result_path=None):
    """
    Makes the prediction and writes the results to the file in the submission format. 
    
    :param model: A model with at least the predict function
    :param threshold: A number between 0 and 1
    
    :returns A set of correlated feature names
    """
    predictions = model.predict(test_data)
    test_data["damage_grade"] = predictions
    # Since XGBoost predicts [0, 1, 2] we have to rename back to original names, i.e. [1, 2, 3]
    if isinstance(model, xgboost.XGBClassifier):
        print("XGBoost model was used. Renaming predictions ...")
        test_data["damage_grade"] = test_data["damage_grade"].replace({0: 1,
                                                                       1: 2,
                                                                       2: 3})
    test_data.to_csv(result_path, columns=["damage_grade"], index_label="building_id")
    print(f"Saved final prediction in '{result_path}'")
