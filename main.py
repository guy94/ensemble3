from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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


def print_best_params(best_params, file_name, model_name):
    """
    print the selected values of each hyper-param
    :param best_params:
    :return:
    """

    print(f"\n\n{model_name} Best Params For Data Set: {file_name.split('/')[1]}")
    for key, val in best_params.items():
        print(f'{key}: {val}')


def run_adaboost(grid_search=False):
    """
    run AdaBoostRegressor model,
    use DecisionTreeRegressor as base estimator,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using RMSE Score metric.
    :return:
    """

    n_estimators = [50, 100, 200]
    learning_rate = [0.03, 0.1, 0.2, 0.5, 1, 1.5]

    total_data = iterate_files()
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=1))
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
            dtc_gs = GridSearchCV(model, params, cv=5).fit(prepared_data.x_train, np.ravel(prepared_data.y_train))

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name, 'AdaBoostRegressor')
            model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=1),
                                      n_estimators=best_params['n_estimators'],
                                      learning_rate=best_params['learning_rate'])

        model.fit(prepared_data.x_train, np.ravel(prepared_data.y_train))
        y_prediction = model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_lightgbm(grid_search=False):
    """
    run LGBMRegressor model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using RMSE Score metric.
    :return:
    """
    n_estimators = [100, 200]
    num_leaves = [5, 10, 20, 31]
    max_depth = [-1, 10, 30, 50]
    boosting_type = ['gbdt', 'dart']

    model = LGBMRegressor()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'num_leaves': num_leaves,
                      'boosting_type': boosting_type}
            dtc_gs = GridSearchCV(model, params, cv=5).fit(prepared_data.x_train, np.ravel(prepared_data.y_train))

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name, 'LGBMRegressor')
            model = LGBMRegressor(n_estimators=best_params['n_estimators'],
                                  max_depth=best_params['max_depth'],
                                  num_leaves=best_params['num_leaves'],
                                  boosting_type=best_params['boosting_type'])

        model.fit(prepared_data.x_train, np.ravel(prepared_data.y_train))
        y_prediction = model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_catboost(grid_search=False):
    """
    run CatBoostRegressor model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using RMSE Score metric.
    :return:
    """

    depth = [6, 8, 10]
    learning_rate = [0.01, 0.05, 0.1]
    iterations = [30, 50, 100]

    model = CatBoostRegressor(logging_level='Silent')
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        if grid_search:
            params = {'depth': depth,
                      'iterations': iterations,
                      'learning_rate': learning_rate}
            dtc_gs = GridSearchCV(model, params, cv=5, scoring='f1_weighted').fit(prepared_data.x_train,
                                                                                  prepared_data.y_train)

            best_params = dtc_gs.best_params_
            print_best_params(best_params, name, 'CatBoostRegressor')
            model = CatBoostRegressor(logging_level='Silent', depth=best_params['depth'],
                                      learning_rate=best_params['learning_rate'],
                                      iterations=best_params['iterations'])

        model.fit(prepared_data.x_train, prepared_data.y_train)
        y_prediction = model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])

    return scores


def evaluate(y_test, y_pred):
    """
    apply RMSE Score metric
    :param y_test:
    :param y_pred:
    :return:
    """
    score = mean_squared_error(y_test, y_pred, squared=False)
    return round(score, 3)


if __name__ == '__main__':

    for i in tqdm(range(1)):
        # catboost_score = run_catboost()
        # print(f'Catboost RMSE Without Hyper-Parameter Tuning: {catboost_score}\n')
        # adaboost_scores = run_adaboost()
        # print(f'Adaboost RMSE Without Hyper-Parameter Tuning: {adaboost_scores}')
        # lightgbm_score = run_lightgbm()
        # print(f'Lightgbm RMSE Without Hyper-Parameter Tuning: {lightgbm_score}')

        catboost_score = run_catboost(grid_search=True)
        print(f'\n\nCatboost RMSE With Hyper-Parameter Tuning: {catboost_score}')
        adaboost_score = run_adaboost(grid_search=True)
        print(f'Adaboost RMSE With Hyper-Parameter Tuning: {adaboost_score}')
        lightgbm_score = run_lightgbm(grid_search=True)
        print(f'Lightgbm RMSE With Hyper-Parameter Tuning: {lightgbm_score}')
