from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import read_csv as reader
from data_preparation import DataPreparation
from models import DTC, ourRandomForest, ourRotationForest
from sklearn.metrics import f1_score


def iterate_files():
    """
    for each csv file - read, prepare, split for train and test sets.
    :return:
    """

    file_names = reader.read_file_names('data')
    total_data = {}
    for name in file_names:
        dp = DataPreparation()
        dp.df = reader.read_file(name)
        prepare_data(dp)
        total_data[name] = dp

    return total_data


def prepare_data(dp):
    """
    calls all relevant functions from data_preparation class
    :param dp:
    :return:
    """

    dp.partition_data_sets()
    dp.fill_na()
    for i in range(len(dp.all_data_frames)):
        dp.discretization(dp.all_data_frames[i])


def run_decision_tree_model(grid_search=False):
    """
    run DecisionTreeClassifier model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using F1 Score metric.
    :return:
    """

    total_data = iterate_files()
    class_model = DTC()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'max_depth': [5, 10, 12, 15, 20, 30, 50], 'min_samples_split': [2, 3, 5, 7]}
            dtc_gs = GridSearchCV(class_model.model, params, cv=5).fit(prepared_data.x_train, prepared_data.y_train)

            best_params = dtc_gs.best_params_
            class_model.model = DTC(max_depth=best_params['max_depth'],
                                    min_samples_split=best_params['min_samples_split'])

        class_model.fit(prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_random_forest(grid_search=False):
    """
    run RandomForestClassifier model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using F1 Score metric.
    :return:
    """
    n_estimators = [100, 200]
    # Number of features to consider at every split
    max_features = ['sqrt', None]
    # Maximum number of levels in tree
    max_depth = [10, 30, 50, None]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    # Class weights
    class_weight = [None, 'balanced']

    class_model = ourRandomForest()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                      'class_weight': class_weight}
            dtc_gs = GridSearchCV(class_model.model, params, cv=5).fit(prepared_data.x_train, prepared_data.y_train)

            best_params = dtc_gs.best_params_
            class_model.model = ourRandomForest(n_estimators=best_params['n_estimators'],
                                                max_depth=best_params['max_depth'],
                                                min_samples_split=best_params['min_samples_split'],
                                                min_samples_leaf=best_params['min_samples_leaf'],
                                                max_features=best_params['max_features'])

        class_model.fit(prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_rotation_forest(grid_search=False):
    """
    run RotationTreeClassifier model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using F1 Score metric.
    :return:
    """

    n_features_per_subset = [3, 5]
    # Number of features to consider in each classifier
    max_depth = [10, 30, 50, None]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    class_weight = [None, 'balanced']

    class_model = ourRotationForest()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_features_per_subset': n_features_per_subset, 'max_depth': max_depth,
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                      'class_weight': class_weight}
            dtc_gs = GridSearchCV(class_model.model, params, cv=5).fit(prepared_data.x_train, prepared_data.y_train)

            best_params = dtc_gs.best_params_
            class_model.model = ourRotationForest(n_features_per_subset=best_params['n_features_per_subset'],
                                                 max_depth=best_params['max_depth'],
                                                 min_samples_split=best_params['min_samples_split'],
                                                 min_samples_leaf=best_params['min_samples_leaf'])

        class_model.fit(prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])

    return scores


def evaluate(y_test, y_pred):
    """
    apply F1 Score metric.
    considers both Recall and Precision metrics.
    :param y_test:
    :param y_pred:
    :return:
    """

    return f1_score(y_test, y_pred, average=None)


if __name__ == '__main__':

    for i in tqdm(range(1)):

        decision_tree_scores = run_decision_tree_model()
        print(f'Decision Tree F1 Score Without HyperParameters Tuning: {decision_tree_scores}')
        decision_tree_scores = run_decision_tree_model(grid_search=True)
        print(f'Decision Tree F1 Score With HyperParameters Tuning: {decision_tree_scores}\n')

        random_forest_score = run_random_forest()
        print(f'Random Forest F1 Score Without HyperParameters Tuning: {random_forest_score}')
        random_forest_score = run_random_forest(grid_search=True)
        print(f'Random Forest F1 Score With HyperParameters Tuning: {random_forest_score}\n')

        rotation_forest_score = run_rotation_forest()
        print(f'Rotation Forest F1 Score Without HyperParameters Tuning: {rotation_forest_score}')
        rotation_forest_score = run_rotation_forest(grid_search=True)
        print(f'Rotation Forest F1 Score With HyperParameters Tuning: {rotation_forest_score}')
