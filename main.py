from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import read_csv as reader
from data_preparation import DataPreparation
from models import DTC, ourRandomForest, ourRotationForest
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns


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
        prepare_data(dp, name)
        total_data[name] = dp

    return total_data


def prepare_data(dp, file_name):
    """
    calls all relevant functions from data_preparation class
    :param dp:
    :return:
    """

    dp.partition_data_sets()
    dp.fill_na()
    correlation(dp, file_name)
    for i in range(len(dp.all_data_frames)):
        dp.discretization(dp.all_data_frames[i])


def correlation(dp, file_name):
    """
    plot correlation heatmap of the features of each data set.
    :param dp:
    :return:
    """
    plt.figure(figsize=(14, 10))
    plt.title(file_name.split('/')[1].split('.')[0])
    cor = dp.df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def print_best_params(best_params, name):
    """
    print the selected values of each hyper-param
    :param best_params:
    :return:
    """

    print(f"\n\nDecision Tree Best Params For Data Set: {name.split('/')[1]}")
    for key, val in best_params.items():
        print(f'{key}: {val}')


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
            dtc_gs = GridSearchCV(class_model.model, params, cv=5).fit(prepared_data.x_train, np.ravel(prepared_data.y_train))

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name)
            class_model = DTC(max_depth=best_params['max_depth'],
                                    min_samples_split=best_params['min_samples_split'])

        class_model.fit(prepared_data.x_train, np.ravel(prepared_data.y_train))
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

    class_model = ourRandomForest()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
            dtc_gs = GridSearchCV(class_model.model, params, cv=5).fit(prepared_data.x_train, np.ravel(prepared_data.y_train))

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name)
            class_model = ourRandomForest(n_estimators=best_params['n_estimators'],
                                                max_depth=best_params['max_depth'],
                                                min_samples_split=best_params['min_samples_split'],
                                                min_samples_leaf=best_params['min_samples_leaf'],
                                                max_features=best_params['max_features'])

        class_model.fit(prepared_data.x_train, np.ravel(prepared_data.y_train))
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
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
            dtc_gs = GridSearchCV(class_model.model, params, cv=5, scoring='f1_weighted').fit(prepared_data.x_train, prepared_data.y_train)

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name)
            class_model = ourRotationForest(n_features_per_subset=best_params['n_features_per_subset'],
                                                 max_depth=best_params['max_depth'],
                                                 min_samples_split=best_params['min_samples_split'],
                                                 min_samples_leaf=best_params['min_samples_leaf'])

        class_model.fit(prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])

    return scores


def evaluate(y_test, y_pred):
    """
    apply F1 Score metric, calculating the weighted average of all classes classifications.
    considers both Recall and Precision metrics.
    :param y_test:
    :param y_pred:
    :return:
    """
    score = f1_score(y_test, y_pred, average='weighted')
    return round(score, 3)


if __name__ == '__main__':

    for i in tqdm(range(1)):

        # decision_tree_scores = run_decision_tree_model()
        # print(f'\nDecision Tree F1 Score Without Hyper-Parameter Tuning: {decision_tree_scores}')
        # random_forest_score = run_random_forest()
        # print(f'Random Forest F1 Score Without Hyper-Parameter Tuning: {random_forest_score}')
        # rotation_forest_score = run_rotation_forest()
        # print(f'Rotation Forest F1 Score Without Hyper-Parameter Tuning: {rotation_forest_score}\n')

        decision_tree_scores = run_decision_tree_model(grid_search=True)
        print(f'Decision Tree F1 Score With Hyper-Parameter Tuning: {decision_tree_scores}')
        random_forest_score = run_random_forest(grid_search=True)
        print(f'Random Forest F1 Score With Hyper-Parameter Tuning: {random_forest_score}')
        rotation_forest_score = run_rotation_forest(grid_search=True)
        print(f'Rotation Forest F1 Score With Hyper-Parameter Tuning: {rotation_forest_score}')
