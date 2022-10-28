from pyexpat import model
import numpy as np
import pandas as pd
import coremltools as ct
from sklearn.model_selection import train_test_split
import cfbd
from multiprocessing.pool import ThreadPool
np.set_printoptions(precision=3, suppress=True)
import logging
from tqdm import tqdm
class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from tqdm import tqdm
from secret import key

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = key
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_config = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_config)
rankings_api = cfbd.RankingsApi(api_config)
games_api = cfbd.GamesApi(api_config)
stats_api = cfbd.StatsApi(api_config)
betting_api = cfbd.BettingApi(api_config)
metrics_api = cfbd.MetricsApi(api_config)
ratings_api = cfbd.RatingsApi(api_config)
rectruiting_api = cfbd.RecruitingApi(api_config)
conferences_api = cfbd.ConferencesApi(api_config)

def get_data(state=42):
    cfb_train = pd.read_csv('merged_dataset.csv')
    cfb_features = cfb_train.copy()
    cfb_labels = pd.concat([cfb_features.pop(x) for x in ['home_points', 'away_points']], axis=1)
    #Remove Features
    removal_list = ['game_id', 'home_team', 'away_team', 'home_games', 'away_games', 'home_penalties', 'away_penalties',
                    'home_penaltyYards', 'away_penaltyYards','home_defense_havoc_front_seven', 
                    'away_defense_havoc_front_seven', 'home_defense_passing_plays_total_ppa', 
                    'away_defense_passing_plays_total_ppa', 'home_offense_havoc_front_seven', 
                    'away_offense_havoc_front_seven']
    unused = pd.concat([cfb_features.pop(x) for x in removal_list], axis=1)
    cfb_features.head()
    cfb_features = np.array(cfb_features)
    cfb_labels = np.array(cfb_labels)
    cfb_features_train, cfb_features_test, cfb_labels_train, cfb_labels_test = train_test_split(cfb_features, 
                                                                                            cfb_labels, 
                                                                                            test_size=0.33, 
                                                                                            random_state=state)
    return cfb_features_train, cfb_features_test, cfb_labels_train, cfb_labels_test

def get_rse():
    record = games_api.get_team_records(year=2022)
    stats = stats_api.get_team_season_stats(year=2022)
    elo = ratings_api.get_elo_ratings(year=2022)
    return record, stats, elo

def get_lines(year, week):
    conference_list = ["American Athletic", "ACC", "Big 12", "Big Ten", "Conference USA", "FBS Independents",
                  "Mid-American", "Mountain West", "Pac-12", "SEC","Sun Belt"]
    return [line for line in betting_api.get_lines(year=2022)
         if line.away_conference in conference_list and 
         line.home_conference in conference_list 
         and line.week<week]

def get_games(year, week):
    return [game for game in games_api.get_games(year=year, week=week) if game.away_division == 'fbs' and game.home_division == 'fbs']

def get_stats_list():
    l = []
    for x in range(2004,2022):
        k = stats_api.get_team_season_stats(year=x, team='Michigan')
        k_tot = sorted([l.stat_name for l in k])
        l.append(k_tot)
    return list(set.intersection(*map(set,l)))

def get_game_data(elo, home_team, away_team):  
    home_elo = 0
    away_elo = 0
    for team in elo:
        if team.team==home_team:
            home_elo = team.elo
        if team.team==away_team:
            away_elo = team.elo
    home_points = 0
    away_points = 0
    home_team = home_team
    away_team = away_team
    game_id = 0
    line = [game_id, home_team, away_team, home_points, away_points, home_elo, away_elo]
    header = ['game_id', 'home_team', 'away_team', 'home_points', 'away_points', 'home_elo', 'away_elo']
    return line, header

def get_talent_data(talent, home_team, away_team):
    header = ['home_talent', 'away_talent']
    line = [None, None]
    for team in talent:
        if team.school == home_team:
            line[0] = team.talent
        if team.school == away_team:
            line[1] = team.talent
    return line, header

def get_record_data(record, home_team, away_team):
    for school in record:
        if school.team == home_team:
            home_record = school
    for school in record:
        if school.team == away_team:
            away_record = school
    home_games = home_record.total.games
    home_wp = home_record.total.wins/home_record.total.games
    away_games = away_record.total.games
    away_wp = away_record.total.wins/away_record.total.games
    line = [home_games, home_wp, away_games, away_wp]
    header = ['home_games', 'home_wp', 'away_games', 'away_wp'] 
    return line, header

def get_stat_data(stats, home_team, away_team, stats_list):
    line = []
    header = []
    home_season_stats = []
    away_season_stats = []
    for stat in stats:
        if stat.team == home_team:
            if stat.stat_name == 'games':
                continue
            home_season_stats.append(stat)
        if stat.team == away_team:
            if stat.stat_name == 'games':
                continue
            away_season_stats.append(stat)
    h = {}
    a = {}
    if len(home_season_stats)<27 or len(away_season_stats)<27:
        return False, False
    for i,stat in enumerate(home_season_stats):
        if stat.stat_name in stats_list:
            h[stat.stat_name] = stat.stat_value
    for i,stat in enumerate(away_season_stats):
        if stat.stat_name in stats_list:
            a[stat.stat_name] = stat.stat_value
    for key,value in sorted(h.items()):
        header.append('home_'+key)
        line.append(value)
    for key,value in sorted(a.items()):
        header.append('away_'+key)
        line.append(value)
    return line, header

def get_metrics_data(year, home_team, away_team):
    line = []
    header = []
    home_advanced_season_stats = stats_api.get_advanced_team_season_stats(year=year, team=home_team)
    away_advanced_season_stats = stats_api.get_advanced_team_season_stats(year=year, team=away_team)
    if len(home_advanced_season_stats)==0 or len(away_advanced_season_stats)==0:
        return False, False
    home_advanced_season_stats = home_advanced_season_stats[0]
    away_advanced_season_stats = away_advanced_season_stats[0]
    hass = home_advanced_season_stats.to_dict()
    aass = away_advanced_season_stats.to_dict()
    
    new_line = []
    new_header = []
    side = ['defense','offense']
    for s in side:
        for key,value in sorted(hass[s].items()):
            if type(value) != dict:
                new_header.append("home_"+s+"_"+key)
                new_line.append(value)
            else:
                for key2,value2 in sorted(hass[s][key].items()):
                    new_header.append("home_"+s+"_"+key+"_"+key2)
                    new_line.append(value2)
    line.extend(new_line)
    header.extend(new_header)
    
    new_line = []
    new_header = []
    side = ['defense','offense']
    for s in side:
        for key,value in sorted(aass[s].items()):
            if type(value) != dict:
                new_header.append("away_"+s+"_"+key)
                new_line.append(value)
            else:
                for key2,value2 in sorted(aass[s][key].items()):
                    new_header.append("away_"+s+"_"+key+"_"+key2)
                    new_line.append(value2)
    line.extend(new_line)
    header.extend(new_header)
    return line, header

def make_df(record, stats, elo, stats_list, home_team, away_team):
    curr_line = []
    curr_header = []
    #add game data
    game_line, game_header = get_game_data(elo, home_team, away_team)
    curr_line.extend(game_line)
    curr_header.extend(game_header)

    #add record data
    record_line, record_header = get_record_data(record, home_team, away_team)
    curr_line.extend(record_line)
    curr_header.extend(record_header)

    #add season stat data
    stat_line, stat_header = get_stat_data(stats, home_team, away_team, stats_list)
    curr_line.extend(stat_line)
    curr_header.extend(stat_header)

    #add advanced metrics data
    metrics_line, metrics_header = get_metrics_data(2022, home_team, away_team)
    curr_line.extend(metrics_line)
    curr_header.extend(metrics_header)
    df = pd.DataFrame([curr_line], columns=curr_header)
    cols = list(df.columns)
    nc = []
    c = []
    for i,x in enumerate(cols):
        if i<11:
            nc.append(x)
        if 'total' in x:
            c.append(x)
        if 'drives' in x:
            c.append(x)
        if 'rate' in x:
            nc.append(x)
        if 'average' in x:
            nc.append(x)
        if 'explosiveness' in x:
            nc.append(x)
    new_cols = []
    for x in cols:
        if x not in nc and x not in c:
            new_cols.append(x)
    c.extend(new_cols[:50])
    new_cols = new_cols[50:]
    ac = 0
    hc = 0
    for i,x in enumerate(new_cols):
        if 'home' in x:
            hc+=1
        if 'away' in x:
            ac+=1
    for i,x in enumerate(new_cols):
        if 'havoc' in x:
            nc.append(x)
        if '_defense_plays' in x:
            c.append(x)
        if '_offense_plays' in x:
            c.append(x)
    for x in nc:
        if x in new_cols:
            new_cols.remove(x)
    for x in c:
        if x in new_cols:
            new_cols.remove(x)
    for i,x in enumerate(new_cols):
        nc.append(x)
    new_cols = []
    for x in c:
        if 'home' in x:
            df[x] = df[x]/df['home_games']
        if 'away' in x:
            df[x] = df[x]/df['away_games']
    df_features = df.copy()
    df_labels = pd.concat([df_features.pop(x) for x in ['home_points', 'away_points']], axis=1)
    #Remove Features
    removal_list = ['game_id', 'home_team', 'away_team', 'home_games', 'away_games', 'home_penalties', 'away_penalties',
                    'home_penaltyYards', 'away_penaltyYards','home_defense_havoc_front_seven', 
                    'away_defense_havoc_front_seven', 'home_defense_passing_plays_total_ppa', 
                    'away_defense_passing_plays_total_ppa', 'home_offense_havoc_front_seven', 
                    'away_offense_havoc_front_seven']
    unused = pd.concat([df_features.pop(x) for x in removal_list], axis=1)
    df_features.head()
    df_features = np.array(df_features)
    df_labels = np.array(df_labels)
    return df_features

def get_team_data(home_team, away_team):
    with open("team_attributes2.csv", "r") as read:
        csv_reader = csv.reader(read)
        teams = list(csv_reader)[1:]
    read.close()
    home = []
    away = []
    for team in teams:
        if team[0]==home_team:
            home = team
        if team[0]==away_team:
            away = team  
    add = [home[1]]
    add.extend([away[1]])
    add.extend([home[2]])
    add.extend([away[2]])
    add.extend(home[3:27])
    add.extend(away[3:27])
    add.extend(home[27:])
    add.extend(away[27:])
    add = [float(x) for x in add]
    return add

def run_test(test_features, test_labels, model_list):
    correct = 0
    for i,x in enumerate(test_features):
        inp = x.reshape(-1,len(x))
        result = []
        for x in model_list:
            if str(type(x))=="<class \'coremltools.models.model.MLModel\'>":
                result.append(x.predict({'input': inp})['Identity'][0])
            elif str(type(x))=="<class \'keras.engine.sequential.Sequential\'>":
                result.append(x(inp)[0])
            else:
                result.append(x.predict(inp)[0])
        p_h = 0
        for x in result:
            p_h+=x[0]
        p_v = 0
        for x in result:
            p_v+=x[1]
        p_h = p_h/len(result)
        p_v = p_v/len(result)
        h = test_labels[i][0]
        v = test_labels[i][1]
        if h>v and p_h>p_v:
            correct+=1
        if h<v and p_h<p_v:
            correct+=1
    print(correct,"/",test_features.shape[0])
    print("acc = ", correct/test_features.shape[0])
    if len(model_list)==1 and type(model_list[0])!=ct.models.model.MLModel:
        if correct/test_features.shape[0]>0.766:
            model.save("models/acc"+str(correct/test_features.shape[0][:3])+"/model")         

def run(home_team, away_team, model_list):
    inp = np.array([get_team_data(home_team, away_team)])
    if inp.shape[1]==202:
        result = []
        for x in model_list:
            if str(type(x))=="<class \'coremltools.models.model.MLModel\'>":
                result.append(x.predict({'input': inp})['Identity'][0])
            elif str(type(x))=="<class \'keras.engine.sequential.Sequential\'>":
                result.append(x(inp)[0])
            else:
                result.append(x.predict(inp)[0])
        # if type(model_list[0])==ct.models.model.MLModel:
        #     result = [x.predict({'input': inp})['Identity'][0] for x in model_list]
        # else:
        #     result = [x(inp)[0] for x in model_list]
    else:
        print("Unable to test: "+home_team+" vs. "+away_team)
        return False, False
    home = 0
    for x in result:
        home+=x[0]
    away = 0
    for x in result:
        away+=x[1]
    home = home/len(result)
    away = away/len(result)
    return float(home), float(away)

def run_multiple(games, model_list):
    correct = 0
    total = 0
    for game in games:
        ht, at = game.home_team, game.away_team
        hp, ap = game.home_points, game.away_points
        if hp == None or ap == None:
            print(ht+" vs. "+at+" is not final.")
            continue
        php, pap = run(game.home_team, game.away_team, model_list)
        if php == False and pap == False:
            continue
        message = " was incorrect"
        if php>pap and hp>ap:
            correct+=1
            message = ""
        if php<pap and hp<ap:
            correct+=1
            message = ""
        total+=1
        print(game.home_team+": "+str(int(php))+", "+
            game.away_team+": "+str(int(pap))+message)
    print(correct/total)

def predict_multiple(games, model_list):
    for game in games:
        ht, at = game.home_team, game.away_team
        php, pap = run(game.home_team, game.away_team, model_list)
        if php == False or pap == False:
            print("Unable to test "+ht+" vs. "+at)
            continue
        print(ht+": "+str(int(php))+", "+at+": "+str(int(pap)))

def test_on_spread(lines, model_list):
    correct = 0
    spread_correct = 0
    total = 0
    for line in lines:
        ht, at = line.home_team, line.away_team
        hp, ap = line.home_score, line.away_score
        php, pap = run(ht, at, model_list)
        if php>pap and hp>ap:
            correct+=1
        if php<pap and hp<ap:
            correct+=1
            
        provider = None
        if len(line.lines)==0:
            continue
        for x in line.lines:
            if x.provider == 'consensus':
                provider = x
        if provider == None:
            provider = line.lines[0]
            
        spread = provider.formatted_spread
        spread = spread.split(" ")
        num = abs(float(spread[len(spread)-1]))
        if len(spread)>2:
            favorite = spread[0]+" "+spread[1]
        else:
            favorite = spread[0]
        if favorite == ht:
            favorite = 'h'
        else:
            favorite = 'a'
        spread_correct+=check_spread(favorite, line.home_score, line.away_score, php, pap, num)
        total+=1
    print(correct/total)
    print(spread_correct/total)

def check_spread(favorite, hp, ap, php, pap, spread):
    cs = hp-ap
    cas = False
    pcs = php-pap
    pcas = False
    if favorite == 'h':
        if cs<0:
            cas = False
        else:
            if cs>spread:
                cas = True
            else:
                cas = False
                
        if pcs<0:
            pas = False
        else:
            if pcs>spread:
                pcas = True
            else:
                pcas = False
    else:
        if cs>0:
            cas = False
        else:
            if abs(cs)>spread:
                cas = True
            else:
                cas = False
        
        if pcs>0:
            pas = False
        else:
            if abs(pcs)>spread:
                pcas = True
            else:
                pcas = False
    if cas==pcas:
        return 1
    else:
        return 0

def predict_matchup(home_team, away_team, model_list):
    php, pap = run(home_team, away_team, model_list)
    print(home_team+": "+str(int(php))+", "+away_team+": "+str(int(pap)))