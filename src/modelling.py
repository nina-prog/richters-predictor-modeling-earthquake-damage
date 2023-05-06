from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def hyperparameter_optimization(model=None, hyperparameter_grid=None, train_data=None, train_labels=None):
    if model == "Dummy":
        model = DummyClassifier(strategy="most_frequent")

    # TODO: Delete this later in week 04 subtasks
    print("Using RandomForest ...")
    model = RandomForestClassifier(random_state=42)

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
