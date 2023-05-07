from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import matthews_corrcoef, make_scorer

import numpy as np


def hyperparameter_optimization(model=None, hyperparameter_grid=None, train_data=None, train_labels=None, scoring=None):
    if model == "Dummy":
        model = DummyClassifier(strategy="most_frequent")
    elif model == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    
    # TODO: Delete this later in week 04 subtasks
    print("Using DecisionTree ...")
    
    # Only for evaluating the engineered features, change in the week after
    if scoring == "MCC":
        scoring = make_scorer(matthews_corrcoef)
    elif scoring == "ACC":
        scoring = "accuracy"
        
    model = DecisionTreeClassifier()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_validate(model, train_data, train_labels["damage_grade"].ravel(), cv=10,
                                scoring=scoring,
                                n_jobs=-1,
                                return_train_score=True)
    model.fit(train_data, train_labels)

    print("")
    print(f"CV Training: {round(np.mean(cv_results['train_score']), 4)} +/- {round(np.std(cv_results['train_score']), 4)}")
    print(f"CV Test: {round(np.mean(cv_results['test_score']), 4)} +/- {round(np.std(cv_results['test_score']), 4)}")
    print("")
    
    return model


def make_prediction(model=None, test_data=None, result_path=None):
    """
    Makes the prediction and writes the results to the file in the submission format. 
    
    :param model: A model with at least the predict function
    :param threshold: A number between 0 and 1
    
    :returns A set of correlated feature names
    """
    predictions = model.predict(test_data)
    test_data["damage_grade"] = predictions[:,1]
    test_data.to_csv(result_path, columns=["damage_grade"], index_label="building_id")
