from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import numpy as np
import read_csv as reader
from data_preparation import DataPreparation
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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
    # correlation(dp, file_name)
    dp.discretization(dp.x_train)
    dp.discretization(dp.x_test)


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


def run_adaboost(grid_search=False):
    """
    run AdaBoostClassifier model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using MSE Score metric.
    :return:
    """

    total_data = iterate_files()
    model = AdaBoostClassifier()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'max_depth': [5, 10, 12, 15, 20, 30, 50], 'min_samples_split': [2, 3, 5, 7]}
            dtc_gs = GridSearchCV(model, params, cv=5).fit(prepared_data.x_train, np.ravel(prepared_data.y_train))

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name)
            model = AdaBoostClassifier()

        model.fit(prepared_data.x_train, np.ravel(prepared_data.y_train))
        y_prediction = model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_lightgbm(grid_search=False):
    """
    run LGBMRegressor model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using MSE Score metric.
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

    model = LGBMRegressor()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
            dtc_gs = GridSearchCV(model, params, cv=5).fit(prepared_data.x_train, np.ravel(prepared_data.y_train))

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name)
            model = LGBMRegressor(n_estimators=best_params['n_estimators'],
                                                max_depth=best_params['max_depth'],
                                                min_samples_split=best_params['min_samples_split'],
                                                min_samples_leaf=best_params['min_samples_leaf'],
                                                max_features=best_params['max_features'])

        model.fit(prepared_data.x_train, np.ravel(prepared_data.y_train))
        y_prediction = model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_catboost(grid_search=False):
    """
    run CatBoostRegressor model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using MSE Score metric.
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

    model = CatBoostRegressor()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_features_per_subset': n_features_per_subset, 'max_depth': max_depth,
                      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
            dtc_gs = GridSearchCV(model, params, cv=5, scoring='f1_weighted').fit(prepared_data.x_train, prepared_data.y_train)

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name)
            class_model = CatBoostRegressor(n_features_per_subset=best_params['n_features_per_subset'],
                                                 max_depth=best_params['max_depth'],
                                                 min_samples_split=best_params['min_samples_split'],
                                                 min_samples_leaf=best_params['min_samples_leaf'])

        model.fit(prepared_data.x_train, prepared_data.y_train)
        y_prediction = model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])

    return scores


def evaluate(y_test, y_pred):
    """
    apply MSE Score metric
    :param y_test:
    :param y_pred:
    :return:
    """
    score = mean_squared_error(y_test, y_pred, multioutput="uniform_average")
    return round(score, 3)


if __name__ == '__main__':

    for i in tqdm(range(1)):
        adaboost_scores = run_adaboost()
        print(f'Adaboost MSE Without Hyper-Parameter Tuning: {adaboost_scores}')
        lightgbm_score = run_lightgbm()
        print(f'Lightgbm MSE Without Hyper-Parameter Tuning: {lightgbm_score}')
        catboost_score = run_catboost()
        print(f'Catboost MSE Without Hyper-Parameter Tuning: {catboost_score}\n')

        adaboost_scores = run_adaboost(grid_search=True)
        print(f'Adaboost MSE With Hyper-Parameter Tuning: {adaboost_scores}')
        lightgbm_score = run_lightgbm(grid_search=True)
        print(f'Lightgbm MSE With Hyper-Parameter Tuning: {lightgbm_score}')
        catboost_score = run_catboost(grid_search=True)
        print(f'Catboost MSE With Hyper-Parameter Tuning: {catboost_score}')
