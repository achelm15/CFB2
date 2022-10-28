import coremltools as ct
import tensorflow as tf
from tensorflow import keras
from ml.cfb_utils import (get_games, predict_matchup, predict_multiple)
import numpy as np
import argparse
import pickle
np.set_printoptions(precision=3, suppress=True)

def main(args):
    home = args.home
    away = args.away
    week = args.week
    if (not home or not away) and week == False:
        print("Enter both a home and away team")
        return
    ml = args.ml
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
    if week:
        games = get_games(2022, week)
        predict_multiple(games, model_list)
    else:
        predict_matchup(home, away, model_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='College Football Predictions')
    parser.add_argument('--file', default='774', type=str, help='File Path of model')
    parser.add_argument('--home', default=False, type=str, help='Home Team')
    parser.add_argument('--away', default=False, type=str, help='Away Team')
    parser.add_argument('--week', default=False, type=int, help='Enter Week Number to Predict Games')
    parser.add_argument('--ml', default=True, type=bool, help='Use CoreML Model')
    args = parser.parse_args()
    main(args)