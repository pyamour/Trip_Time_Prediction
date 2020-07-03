import pandas as pd
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from Trip_Time_Prediction.config.constants import *

def generate_train_test_data(df):
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['log_trip_time']
    x = df[TRAIN_COLS]
    x_train_all, x_test, y_train_all, y_test = train_test_split(x.values, y.values, test_size=0.1, random_state=42,
                                                                shuffle=True)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42,
                                                          shuffle=True)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def train_xgboost_with_gpu(x_train, y_train, x_test, y_test, xgb_path):
    # TODO: num_round = 100
    num_round = 1
    params = {'objective': 'reg:squarederror',  # Specify multiclass classification
              'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
              'predictor': 'gpu_predictor',
              'eval_metric': 'mae'
              }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    gridsearch_params1 = [
        (max_depth, min_child_weight)
        for max_depth in range(5, 12)
        for min_child_weight in range(1, 8)
    ]
    gridsearch_params2 = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(3, 11)]
        for colsample in [i / 10. for i in range(3, 11)]
    ]
    gridsearch_params3 = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    tmp = time.time()
    max_depth, min_child_weight = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params1,
                                                        ['max_depth', 'min_child_weight'])
    subsample, colsample = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params2,
                                                 ['subsample', 'colsample_bytree'])
    eta = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params3, ['eta'])

    best_params = {
        'tree_method': 'gpu_hist',
        'colsample_bytree': colsample,
        'eta': eta,
        'eval_metric': 'mae',
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'objective': 'reg:squarederror',
        'subsample': subsample
    }
    print(best_params)
    gpu_res = {}
    bst = xgb.train(best_params, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))
    if os.path.exists(xgb_path):
        os.remove(xgb_path)
    bst.save_model(xgb_path)
    return

def train_xgboost_with_cpu(x_train, y_train, x_test, y_test, xgb_path):
    # TODO: num_round = 100
    num_round = 1
    params = {'objective': 'reg:squarederror',  # Specify multiclass classification
              'tree_method': 'hist',
              'predictor': 'cpu_predictor',
              'eval_metric': 'mae'
              }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    gridsearch_params1 = [
        (max_depth, min_child_weight)
        for max_depth in range(5, 12)
        for min_child_weight in range(1, 8)
    ]
    gridsearch_params2 = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(3, 11)]
        for colsample in [i / 10. for i in range(3, 11)]
    ]
    gridsearch_params3 = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    tmp = time.time()
    max_depth, min_child_weight = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params1,
                                                        ['max_depth', 'min_child_weight'])
    subsample, colsample = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params2,
                                                 ['subsample', 'colsample_bytree'])
    eta = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params3, ['eta'])

    best_params = {
        'tree_method': 'hist',
        'colsample_bytree': colsample,
        'eta': eta,
        'eval_metric': 'mae',
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'objective': 'reg:squarederror',
        'subsample': subsample
    }
    print(best_params)
    cpu_res = {}
    bst = xgb.train(best_params, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
    print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
    if os.path.exists(xgb_path):
        os.remove(xgb_path)
    bst.save_model(xgb_path)
    return


def xgb_param_grid_search(params, dtrain, num_boost_round, gridsearch_params, param_list):
    min_mae = float("Inf")
    best_params = None
    if len(param_list) == 2:
        for ele1, ele2 in gridsearch_params:
            print("CV with {}={}, {}={}".format(param_list[0], ele1, param_list[1], ele2))
            # Update our parameters
            params[param_list[0]] = ele1
            params[param_list[1]] = ele2
            # Run CV
            cv_results = xgb_cv(params, dtrain, num_boost_round)
            # Update best MAE
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (ele1, ele2)
        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
        return best_params[0], best_params[1]

    if len(param_list) == 1:
        for eta in gridsearch_params:
            print("CV with eta={}".format(eta))
            params[param_list[0]] = eta
            # Run and time CV
            cv_results = xgb_cv(params, dtrain, num_boost_round)
            # Update best score
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = eta
        print("Best params: {}, MAE: {}".format(best_params, min_mae))

        return best_params

def xgb_cv(params, dtrain, num_boost_round):
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics=['mae'],
        early_stopping_rounds=100
    )
    return cv_results

def train_model_xgb():
    for city in city_list:
        df_x = pd.read_csv(DATA_PATH + '/processed/' + city + '_x_data.csv')
        df_y = pd.read_csv(DATA_PATH + '/processed/' + city + '_y_data.csv')
        df = pd.concat([df_x, df_y], axis=1)
        x_train, y_train, x_valid, y_valid, x_test, y_test = generate_train_test_data(df)
        xgb_path = XGB_PATH + city + '_' + XGB_MODEL_SUFFIX
        #train_xgboost_with_gpu(x_train, y_train, x_test, y_test, xgb_path)
        train_xgboost_with_cpu(x_train, y_train, x_test, y_test, xgb_path)

if __name__ == '__main__':
    train_model_xgb()