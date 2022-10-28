from ml.cfb_utils import (get_data, run_test, get_games, run_multiple, get_lines, test_on_spread)
import numpy as np
import argparse
import coremltools as ct
import pickle
import tensorflow as tf
from tensorflow import keras
np.set_printoptions(precision=3, suppress=True)

def main(args):
    games = args.games
    ml = args.ml
    spread = args.spread
    file_list = args.file
    file_list = file_list.split(' ')
    pre_list = []
    for x in file_list:
        if x == 'elastic':
            with open('ml/scikit/elastic_model.pkl', 'rb') as file:
                pre_list.append(pickle.load(file))
        if x == 'lars':
            with open('ml/scikit/lars_model.pkl', 'rb') as file:
                pre_list.append(pickle.load(file))
        if x == 'lassolars':
            with open('ml/scikit/lassolars_model.pkl', 'rb') as file:
                pre_list.append(pickle.load(file))
        if x == 'ridge':
            with open('ml/scikit/ridge_model.pkl', 'rb') as file:
                pre_list.append(pickle.load(file))
    file_list = [x for x in file_list if x not in ['elastic', 'ridge', 'lars', 'lassolars']]
    if ml:
        model_list = [ct.models.MLModel('ml/coreml_models/acc_'+x+"/model.mlpackage") for x in file_list]
    else:
        model_list = [keras.models.load_model('ml/models/acc_'+x+"/model") for x in file_list]
    model_list.extend(pre_list)
    if spread:
        lines = get_lines(2022, 8)
        test_on_spread(lines, model_list)
    else:
        if games != False:
            games = get_games(2022, games)
            run_multiple(games, model_list)
        else:
            _, cfb_features_test, _, cfb_labels_test = get_data()
            run_test(cfb_features_test, cfb_labels_test, model_list)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='College Football Predictions')
    parser.add_argument('--file', default='774', type=str, help='File Path of model')
    parser.add_argument('--games', default=False, type=int, help='Check Previous Games')
    parser.add_argument('--spread', default=False, type=bool, help='Check Spread')
    parser.add_argument('--ml', default=False, type=bool, help='Use CoreML Model')
    args = parser.parse_args()
    main(args)