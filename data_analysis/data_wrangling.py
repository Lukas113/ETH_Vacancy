# -*- coding: utf-8 -*-

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../database')

from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
import datetime, inspect
from DBAccess import DBAccess
from os import path
import yaml
from ipywidgets import interact, widgets, fixed


def get_config():
    """Gets the config yaml file as dict"""
    dir_path = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
    with open(dir_path + "/../config.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config

def _shift_interv(df_sens):
    """
    Shifts 'sensor_numeric1' and 'sensor_numeric2' accordingly 'corr.csv' to correct the data
    
    :param df_sens: pd.DataFrame, where 'sensor_type' must contain only 'turn'
    
    :returns: pd.DataFrame, corrected `df_sens`
    """
    dir_path = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
    df_corr = pd.read_csv(dir_path + '/../database/corr.csv')
    df_corr = df_corr.astype({'start': 'datetime64', 'end': 'datetime64'})
    for ix in range(df_corr.shape[0]):
        row = df_corr.iloc[ix, :]

        df_sens.loc[(df_sens['sensor'] == row['sensorname']) & (df_sens['sensor_timestamp'] > row['start']) & (df_sens['sensor_timestamp'] < row['end']), 'sensor_numeric1'] = \
        df_sens[(df_sens['sensor'] == row['sensorname']) & (df_sens['sensor_timestamp'] > row['start']) & (df_sens['sensor_timestamp'] < row['end'])]['sensor_numeric1'] + row['shift']

        df_sens.loc[(df_sens['sensor'] == row['sensorname']) & (df_sens['sensor_timestamp'] > row['start']) & (df_sens['sensor_timestamp'] < row['end']), 'sensor_numeric2'] = \
        df_sens[(df_sens['sensor'] == row['sensorname']) & (df_sens['sensor_timestamp'] > row['start']) & (df_sens['sensor_timestamp'] < row['end'])]['sensor_numeric2'] + row['shift']
        
    return df_sens


def rename_person(df_person, *columns):
    if 'sex' in columns:
        df_person.loc[df_person.sex == 'm', 'sex'] = 'Männlich'
        df_person.loc[df_person.sex == 'f', 'sex'] = 'Weiblich'
        df_person = df_person.rename(columns={'sex': 'Geschlecht'})
    if 'single_couple' in columns:
        df_person.loc[df_person.single_couple == 'single', 'single_couple'] = 'Singles'
        df_person.loc[df_person.single_couple == 'couple', 'single_couple'] = 'Paare'
        df_person = df_person.rename(columns={'single_sex_couple': 'Gleichgeschlechtiges_Paar'})
    if 'profession' in columns:
        df_person.loc[df_person.profession == 'Architect', 'profession'] = 'Architekt'
        df_person.loc[df_person.profession == 'Other', 'profession'] = 'Andere'
        df_person = df_person.rename(columns={'profession': 'Beruf'})
    return df_person


def _append_nones_malf(df_sens, move=False):
    """
    Creates None values in `df_sens` at the specified sensors at the start and end of the intervals from 'Malfunction.csv'
    
    :param df_sens: pd.Dataframe, min. required is the log_sensor data
    :param move: boolean, does include moving sensor_logs even if their position is corrupt 
                 (usable == 0 --> not usable, usable == 1 --> usable for moving, but not position purpose)

    :returns: `df_sens` with appended None values
    """
    dir_path = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
    df_malf = pd.read_csv(dir_path + '/../database/Malfunction.csv')
    start, end = 'start', 'end'

    df_malf = df_malf.astype({start: 'datetime64', end: 'datetime64'})
    gap_rec = dict(zip(df_sens.columns, [None] * df_sens.shape[1]))
    for interv in df_malf.iterrows():
        # if position data is usable, skip this interval in 'Malfunctions.csv'
        if move and interv[1]['usable'] == 1:
            pass
        else:
            # create start and end None values in case the malfunction overlaps multiple occupation_periods
            sensor = interv[1]['sensorname']
            gap_rec['sensor'] = sensor
            dt_start = interv[1][start] + datetime.timedelta(seconds=2)
            gap_rec['sensor_timestamp'] = dt_start
            df_sens = df_sens.append(gap_rec, ignore_index=True)

            dt_end = interv[1][end] - datetime.timedelta(seconds=2)
            gap_rec['sensor_timestamp'] = dt_end
            df_sens = df_sens.append(gap_rec, ignore_index=True)

            df_sens = filter_interval(df_sens, 'malfunction', dt_start, dt_end, sensor)

    df_sens = df_sens.sort_values(by='sensor_timestamp')
    df_sens = df_sens.reset_index(drop=True)

    return df_sens


def _append_nones_nooc(df_sens):
    """
    Creats None values in `df_sens` on every 'sensor' in `df_sens` at the start and end of the intervlals from 'not_occupied.csv'

    :param df_sens: pd.DataFrame, min.required is the log_sensor data

    :returns: `df_sens` with appended None values
    """
    dir_path = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
    df_nooc = pd.read_csv(dir_path + '/../database/not_occupied.csv')
    start, end = 'empty_from', 'empty_to'

    df_nooc = df_nooc.astype({start: 'datetime64', end: 'datetime64'})
    gap_rec = dict(zip(df_sens.columns, [None] * df_sens.shape[1]))
    sensors = df_sens.loc[:, 'sensor'].unique()
    for interv in df_nooc.iterrows():
        for sensor in sensors:
            gap_rec['sensor'] = sensor
            dt_start = interv[1][start] + datetime.timedelta(seconds=2)
            gap_rec['sensor_timestamp'] = dt_start
            df_sens = df_sens.append(gap_rec, ignore_index=True)

            dt_end = interv[1][end] - datetime.timedelta(seconds=2)
            gap_rec['sensor_timestamp'] = dt_end
            df_sens = df_sens.append(gap_rec, ignore_index=True)
            
        df_sens = filter_interval(df_sens, 'not_occupied', dt_start, dt_end)
            
    df_sens = df_sens.sort_values(by='sensor_timestamp')
    df_sens = df_sens.reset_index(drop=True)

    return df_sens


def _append_nones_inh(df_sens):
    """
    Creats None values in `df_sens` on every 'sensor' in `df_sens` from 'inhabitants.csv' between each time interval if no test persons are present
    
    :param df_sens: pd.DataFrame, min.required is the log_sensor data
    
    :returns: `df_sens` with appended None values
    """
    dir_path = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
    df_inh = pd.read_csv(dir_path + '/../database/inhabitants.csv')
    start, end = 'start', 'end'

    df_inh = df_inh[df_inh.attendance == 1]
    df_inh = df_inh.groupby(by='occupation_period').first().reset_index()
    df_inh = df_inh.astype({start:'datetime64', end:'datetime64'})
    last_date = max(df_inh.loc[:, end])
    df_inh.loc[:, 'end'] = df_inh.loc[:, 'end'].shift(
        1)  # after the shift, end to start is the non_occupied time interval between each occupation

    gap_rec = dict(zip(df_sens.columns, [None] * df_sens.shape[1]))
    sensors = df_sens.loc[:, 'sensor'].unique()
    first = True
    for interv in df_inh.iterrows():
        if first: # gaps calculation starts from the second row
            first = False
            continue
        for sensor in sensors:
            gap_rec['sensor'] = sensor
            dt_end = interv[1][end] + datetime.timedelta(seconds=2)
            gap_rec['sensor_timestamp'] = dt_end
            df_sens = df_sens.append(gap_rec, ignore_index=True)

            dt_start = interv[1][start] - datetime.timedelta(seconds=2)
            gap_rec['sensor_timestamp'] = dt_start
            df_sens = df_sens.append(gap_rec, ignore_index=True)
            
        df_sens = filter_interval(df_sens, 'inhabitants', dt_end, dt_start)
    
    info = get_config()['general_info']
    project_end = datetime.datetime.strptime(info['Projekt_Ende'], '%Y-%m-%d %H:%M:%S.%f')
    interv[1][start], interv[1][end] = project_end, last_date
    df_sens = filter_interval(df_sens, 'inhabitants', last_date, project_end)

    df_sens = df_sens.sort_values(by='sensor_timestamp')
    df_sens = df_sens.reset_index(drop=True)

    return df_sens

# todo work out how to drop data between occupations with filter_interval function
def filter_interval(df_sens, csv, start, end, sensor = None):
    """
    Filters data points which lie between nones if necessary
    :param end: datetime, end of interval
    :param start: datetime, start of interval
    :param csv: str 'malfunction', 'not_occupied' or 'inhabitants'
    :param df_sens: DataFrame to filter

    :return df_sens: input df without data points between start and end
    """
    if csv == 'malfunction':
        df_sens.drop(
            df_sens[(df_sens['sensor'] == sensor) & (df_sens['sensor_timestamp'] > start) \
                    & (df_sens['sensor_timestamp'] < end)].index,
            inplace=True)

    if csv == 'not_occupied' or csv == 'inhabitants':
        df_sens.drop(
            df_sens[(df_sens['sensor_timestamp'] > start) & (
                        df_sens['sensor_timestamp'] < end)].index,
            inplace=True)

    return df_sens


def append_nones_complete(df_sens, move=False):
    df_sens = _append_nones_malf(df_sens, move)
    df_sens = _append_nones_inh(df_sens)
    df_sens = _append_nones_nooc(df_sens)
    df_sens = _trim_nones(df_sens)

    return df_sens



def _trim_nones(df_sens):
    """
    Trims multiple occurences of Nones when they are in a row
    Keeps the first and the last occurence of each row
    
    :param df_sens: pd.DataFrame, min. required is log_sensor data
    
    :returns: trimmed df_sens
    """
    # create correct grouped index to calc with it correctly in groupby(by = 'sensor')
    df_sens = df_sens.sort_values(by=['sensor', 'sensor_timestamp'])
    df_sens = df_sens.reset_index(drop=True)

    drop_ix = []
    df_sens_grouped = df_sens.groupby(by='sensor')
    for sensor, df_s in df_sens_grouped:
        none_ix = np.array(df_s[df_s['log_sensor_id'].isna()].index)
        if len(none_ix) > 2:
            for i in range(len(none_ix) - 2):
                if (none_ix[i + 2] - none_ix[i + 1]) == 1 and (none_ix[i + 1] - none_ix[i]) == 1:
                    drop_ix.append(none_ix[i + 1])

    df_sens = df_sens.drop(drop_ix)
    df_sens = df_sens.sort_values(by='sensor_timestamp')
    df_sens = df_sens.reset_index(drop=True)
    return df_sens


def _append_logs(df_sens, last_log=False):
    """
    Adds an artificial sensor_log one second before a None pair occurs for each 'sensor' in `df_sens`
    
    :param df_sens: pd.DataFrame, min. required is log_sensor data
    :param last_log: boolean, sets an artificial sensor_log min(project_end, now) at the end of the timeframe if True
    
    :returns: enriched df_sens
    """
    info = get_config()['general_info']
    project_end = datetime.datetime.strptime(info['Projekt_Ende'], '%Y-%m-%d %H:%M:%S.%f')
    j = 0
    # create correct grouped index to calc with it correctly in groupby(by = 'sensor')
    df_sens = df_sens.sort_values(by=['sensor', 'sensor_timestamp'])
    df_sens = df_sens.reset_index(drop=True)  # ensures index is sorted
    df_sens = df_sens.reset_index()  # sets index as a column

    df_sens_grouped = df_sens.groupby(by='sensor')
    for sensor, df_s in df_sens_grouped:
        none_ix = np.array(df_s.index[df_s['log_sensor_id'].isna()])
        for ix in none_ix[::2]:
            if (ix - 1) in df_s.index:  # necessary if the first occurence of None is at the start of the df_s
                ix_timestamp = df_s[df_s.loc[:, 'index'] == ix]['sensor_timestamp'] - datetime.timedelta(seconds=1)
                lag_ix = df_s[df_s.loc[:, 'index'] == (ix - 1)].copy()
                lag_ix.loc[:, 'sensor_timestamp'] = ix_timestamp.values[
                    0]  # .vlaues[0] because ix_timestamp is pd.Series
                df_sens = df_sens.append(lag_ix, ignore_index=True)
                j += 1

        if last_log:
            log = df_s.iloc[-1, :].copy()
            if log.log_sensor_id:
                log['sensor_timestamp'] = min(datetime.datetime.today(), project_end)
                df_sens = df_sens.append(log, ignore_index=True)

    df_sens = df_sens.drop(columns=['index'])
    df_sens = df_sens.sort_values(by='sensor_timestamp')
    df_sens = df_sens.reset_index(drop=True)

    return df_sens


def _cut_lower_upper(df_sens):
    """
    Cuts in config.yml specified lower_upper +/- tolerance from the accroding sensor
    
    :param df_sens: pd.DataFrame, min required log_sensor data
    
    :returns: cutted turn sensor data of df_sens
    """
    ts = get_config()['turn_sensors']
    t_tol = ts['Toleranz']  # +/- tolerance of turn sensors in degrees
    lower_upper = {'Drehschrank': (ts['Minimum'] - t_tol, ts['Maximum_Drehschrank'] + t_tol),
                   'Drehwand': (ts['Minimum'] - t_tol, ts['Maximum_Drehwand'] + t_tol), \
                   'LampeAussenwand': (ts['Minimum'] - t_tol, ts['Maximum_LampeAussenwand'] + t_tol),
                   'LampeDrehwand': (ts['Minimum'] - t_tol, ts['Maximum_LampeDrehwand'] + t_tol)}
    pd.options.mode.chained_assignment = None
    for element in lower_upper.keys():
        df_sens.loc[
            (df_sens.sensor == element) & (df_sens.sensor_numeric1 < lower_upper[element][0]), 'sensor_numeric1'] = \
            lower_upper[element][0]
        df_sens.loc[
            (df_sens.sensor == element) & (df_sens.sensor_numeric1 > lower_upper[element][1]), 'sensor_numeric1'] = \
            lower_upper[element][1]
        df_sens.loc[
            (df_sens.sensor == element) & (df_sens.sensor_numeric2 < lower_upper[element][0]), 'sensor_numeric2'] = \
            lower_upper[element][0]
        df_sens.loc[
            (df_sens.sensor == element) & (df_sens.sensor_numeric2 > lower_upper[element][1]), 'sensor_numeric2'] = \
            lower_upper[element][1]

    return df_sens


def _calculate_rotary_weights(df_sens):
    """
    Calculates the time difference between two sensor sensor events of the same sensor
    Adds an additional 'duration_sensor_position at the end of the dataframe'
    
    :params df_sens: pd.DataFrame, where the rows which should be considered as a gap must consist of Nones, except in 'sensor' and 'sensor_timestamp'
    
    :returns: dict, where the key is the 'sensor' and the value the according pd.DataFrame
    """
    df_sens_grouped = df_sens.groupby(by='sensor')
    df_sens_weights = {}
    for sensor, df_s in df_sens_grouped:
        df_s = df_s.reset_index(drop=True)
        df_s['sensor_timestamp_shift'] = df_s.loc[:, 'sensor_timestamp'].shift(-1)
        duration_sensor_position = (df_s.loc[:, 'sensor_timestamp_shift'] - df_s.loc[:, 'sensor_timestamp']).dt.seconds
        df_s['duration_sensor_position'] = duration_sensor_position / (60 * 60 * 24) # computes weights in float64 days
        df_s = df_s.drop(columns=['sensor_timestamp_shift'])
        df_s = df_s.dropna()
        df_sens_weights[sensor] = df_s

    return df_sens_weights


def living_situation_person(df_person):
    """
    Enriches the iformation from person with the columns: single_couple and single_sex_couple
    - single_couple: Value 'single' if in an occupation period hast just one person, otherwise (two persons) 'couple'
    - single_sex_couple: Value 'True' if a couple is of the same sex. Value 'False', if a pair has different genders. Value 'None', if an occupation period contains single
    
    :param df_person: pd.DataFrame from table person
    
    :returns: enriched df_person with single_couple and single_sex_couple
    """
    # create new df with 'occupation_period_id' and 'single_couple' to provide direct information about whether the paricipants are alone or in pairs
    df_pa = df_person[df_person.loc[:, 'attendance'] == 1].astype(
        {'occupation_period_id': 'int64'})  # pa = persons_attended
    df_pac = df_pa.loc[:, ['occupation_period_id', 'person_id']].groupby(by='occupation_period_id').count().rename(
        columns={'person_id': 'count'}).reset_index()  # pac = persons_attended_couples
    df_pac['single_couple'] = 'single'
    df_pac.loc[df_pac.loc[:, 'count'] >= 2, 'single_couple'] = 'couple'
    df_pac = df_pac.drop(columns=['count'])

    # create a new df with 'occupation_period_id' and 'single_sex_count' to provide information whether a couple has the same sex (2 if yes) or not (1 if no)
    df_pa = df_pa.merge(df_pac, on='occupation_period_id')
    df_pasd = df_pa.loc[:, ['occupation_period_id', 'sex', 'person_id']].groupby(
        by=['occupation_period_id', 'sex']).count().rename(columns={'person_id': 'single_sex_count'}) \
                  .reset_index().loc[:, ['occupation_period_id', 'single_sex_count']].drop_duplicates().reset_index(
        drop=True)  # pasd = person_attended_sex_distribution

    # enriches the original `df_person` with the information about 'single_couple' (single or couple) and 'single_sex_couple' (True if yes, False if no, None if occupation is single)
    df_lsp = df_pa.merge(df_pasd, on='occupation_period_id')
    df_lsp.loc[df_lsp.loc[:, 'single_couple'] == 'single', 'single_sex_count'] = None
    df_lsp.loc[(df_lsp.loc[:, 'single_couple'] == 'couple') & (
            df_lsp.loc[:, 'single_sex_count'] == 1), 'single_sex_count'] = False
    df_lsp.loc[(df_lsp.loc[:, 'single_couple'] == 'couple') & (
            df_lsp.loc[:, 'single_sex_count'] == 2), 'single_sex_count'] = True
    df_lsp = df_lsp.rename(columns={'single_sex_count': 'single_sex_couple'})

    return df_lsp


def merge_log_person(dict_dfs, df_person, category):
    """
    Merges the dfs in `dict_dfs` with `df_person` by 'occupation_period_id' so every sensor_log has information about the specified `category`
    The following categories are allowed:
    
    - age
        - singles (df_person.single_couple = 'single') are treated as they are
        - couples (df_person.single_couple = 'couple') couples that do not belong to the same age category are sorted out
    - sex
        - singles (df_person.single_couple = 'single') are treated as they are
        - couples (df_person.single_couple = 'couple') couples that do not share the same sex category are sorted out
    - living
        - considers the living situation of all participants
    
    :param dict_dfs: dict, where key = 'sensor' and value = pd.DataFrame
    :param df_person: pd.DataFrame from table 'person', where information about 'single_couple' and 'single_sex_couple' must be included
    
    :returns: enriched dict_dfs
    """
    df_person = df_person.copy()
    df_person = df_person[df_person['attendance'] == 1]

    if category == 'age':
        df_person = df_person.sort_values('occupation_period_id')
        occupation_period_id_drop_ix = []
        for i in range(1, df_person.shape[0]):
            if (df_person.iloc[i, :].occupation_period_id == df_person.iloc[i - 1, :].occupation_period_id) and (
                    df_person.iloc[i, :].age != df_person.iloc[i - 1, :].age):
                occupation_period_id_drop_ix.append(df_person.iloc[i, :].occupation_period_id)

        drop_ix = df_person.index[df_person['occupation_period_id'].isin(occupation_period_id_drop_ix)]
        df_person = df_person.drop(drop_ix)
        columns = ['occupation_period_id', 'age', 'single_couple', 'single_sex_couple']

    elif category == 'sex':
        drop_ix = df_person.index[df_person['single_sex_couple'] == False]
        df_person = df_person.drop(drop_ix)
        columns = ['occupation_period_id', 'sex', 'single_couple', 'single_sex_couple']

    elif category == 'living':
        columns = ['occupation_period_id', 'single_couple', 'single_sex_couple']

    df_person = df_person.groupby(by='occupation_period_id').first().reset_index()
    df_person = df_person.loc[:, columns]

    # merges log_sensor information in `dict_dfs` with `df_person`
    for sensor in dict_dfs.keys():
        df_sens = dict_dfs[sensor]
        df_sens = df_sens.merge(df_person, on='occupation_period_id')
        dict_dfs[sensor] = df_sens

    return dict_dfs


def sensor_states(df):
    """
    Gets the sensor states 
    
    :param df: pd.DataFrame from log_sensor
    
    :returns df_sens_numeric: dict where the keys are each 'sensor' and the values a pd.DataFrame of each sensor with `sensor_timestamp`, `variable` and `sensor_numeric`
    """
    id_vars = ['sensor_timestamp']  # add more if you want to keep additional columns
    df_sens_grouped = df.groupby(by='sensor')
    df_sens_numeric = {}
    for sensor, data in df_sens_grouped:
        df_numeric = pd.melt(data, id_vars=id_vars, value_vars=['sensor_numeric1', 'sensor_numeric2']) \
            .rename(columns={'value': 'sensor_numeric'}).sort_values(by=['sensor_timestamp', 'variable']).reset_index(
            drop=True)  # puts sensor_numeric1 and sensor_numeric2 among each other
        df_numeric.loc[df_numeric.variable == 'sensor_numeric2', 'sensor_timestamp'] += datetime.timedelta(
            seconds=1)  # add 1 milisec to be able to plot the data correctly in sns.lineplot
        df_sens_numeric[sensor] = df_numeric

    return df_sens_numeric


def prep_rotary_weights_assignment(df, cat=None):
    """
    Gets the cleaned sensor states in order to plot a weighted histogram
    - Only `sensor_numeric2` has been considered becauses sometimes the previous `sensor_numeric2` does not match the next `sensor_numeric1` and we have no further information about this missmatch
    - Generates a None value in `sensor_numeric2` to calculate the weights appropriately in the intervals of 'Malfunction.csv', 'not_occupied.csv' and 'inhabitants.csv'
    - Limitates the upper and lower boundaries to min/max +/- a specified tolerance defined in config.yml
    - Considers Flags: 0 --> 'Correct data'
    
    :param df: pd.DataFrame from tables 'log_sensor' and 'occupation_period'
    :param cat: str, category 'living', 'sex' or 'age' if sensor data has to be enriched with this information, otherwise None (default)
    
    :returns: dict where the keys are each 'sensor' and the values a pd.DataFrame of each sensor with `sensor_timestamp`, `variable` and `sensor_numeric`
    """
    df_sens = df.copy()
    df_sens = df_sens[(df_sens.flag != 1) & (df_sens.flag != 2) & (df_sens.flag != 3) & (df_sens.flag != 5)] # selects flag 0 and 4

    df_sens = _shift_interv(df_sens)
    df_sens = _cut_lower_upper(df_sens)
    df_sens = _append_nones_malf(df_sens)
    df_sens = _append_nones_nooc(df_sens)
    df_sens = _append_nones_inh(df_sens)
    df_sens = _trim_nones(df_sens)
    df_sens = _append_logs(df_sens, last_log=True)

    rotary_weights = _calculate_rotary_weights(df_sens)

    if cat:
        dba = DBAccess()
        person = dba.select('person')
        df_person = living_situation_person(person)
        rotary_weights = merge_log_person(rotary_weights, df_person, cat)

    return rotary_weights


def prep_rotary_sensors(df):
    """
    Gets the cleaned sensor states in order to plot a line-plot
    - Only `sensor_numeric2` has been considered becauses sometimes the previous `sensor_numeric2` does not match the next `sensor_numeric1` and we have no further information about this missmatch
    - Generates a None value in `sensor_numeric2` to create a gap in the line-plot where we are pretty sure that in the intervals of 'Malfunction.csv' the data is not correct
    - Limitates the upper and lower boundaries to min/max +/- a specified tolerance defined in config.yml
    - Considers Flags: 0 --> 'Correct data', 4 --> 'Values out of defined range' and 5 --> 'Time of event outside experiment'
    
    :param df: pd.DataFrame from log_sensor
    
    :returns: dict where the keys are each 'sensor' and the values a pd.DataFrame of each sensor with `sensor_timestamp`, `variable` and `sensor_numeric`
    """
    df_sens = df.copy()
    df_sens = df_sens[(df_sens.flag != 1) & (df_sens.flag != 2) & (df_sens.flag != 3)]

    df_sens = _shift_interv(df_sens)
    df_sens = _cut_lower_upper(df_sens)
    df_sens = _append_nones_malf(df_sens)
    df_sens = _trim_nones(df_sens)
    df_sens = _append_logs(df_sens, last_log=True)

    # lineplot data needs to have start and end in a vector [start, end, start, end, ...] where one `end` and  `start` pair should have the same value in 'sensor_numeric'
    df_sens_numeric = sensor_states(df_sens)

    # creates a dict of pd.DataFrame for each turn sensor
    # ignores the sensor_numeric1 because we assume that the final state of the sensor is the relevant value if end and the next start value do not match
    for sensor in df_sens_numeric.keys():
        df_sens = df_sens_numeric[sensor]
        df_sens['sensor_numeric_lag'] = df_sens.loc[:, 'sensor_numeric'].shift(1)
        df_sens.loc[(df_sens.variable == 'sensor_numeric1') & (df_sens.sensor_numeric != df_sens.sensor_numeric_lag) & (
            df_sens.sensor_numeric_lag.notna()), 'sensor_numeric'] = df_sens.loc[:, 'sensor_numeric_lag']
        df_sens = df_sens.drop(columns=['sensor_numeric_lag'])
        obsolete_rows = np.array(df_sens.index[(df_sens['sensor_numeric'].isna()) & (df_sens[
                                                                                         'variable'] == 'sensor_numeric2')]) - 1  # gets the index of the previous row generated by sensor_states() of each None-value to delete the obsolete rows
        df_sens = df_sens.drop(obsolete_rows)
        df_sens = df_sens.reset_index(drop=True)

        df_sens_numeric[sensor] = df_sens

    return df_sens_numeric


def get_valid_days(data, sensors):
    """
    Get dictionary with valid days for sensors
    :param data: log_sensor dataframe
    :return days_sensor: dictionary with n days for sensor
    """
    # get number of days
    days_sensor = {}
    for s in sensors:
        weeks = len(data[data.sensor == s][
                        'sensor_timestamp'].dt.date.unique())  # number of occupation_periods in dataset /drehwand
        days_sensor[s] = weeks

    return days_sensor


def group_week_count(data, sensors):
    # data wrangling
    """
    Group sensors by weekday, count movement per day
    :param data: dataframe containing log_sensor table data
    :return clean_wd: dataframe containing amount of movement per sensor and weekday
    """
    days_sensor = get_valid_days(data, sensors)

    clean_wd = data.groupby(data.sensor_timestamp.dt.weekday)
    clean_wd = clean_wd.sensor.apply(pd.value_counts)
    clean_wd = clean_wd.reset_index().rename(columns={'sensor_timestamp': 'weekday',
                                                      'level_1': 'sensor',
                                                      'sensor': 'count'})
    return clean_wd


# todo https://stackoverflow.com/questions/52409712/fill-in-missing-hours-in-a-pandas-dataframe
def group_by_hour(data, sensors):
    # data wrangling
    """
    Groups sensor data by hour and occupation period fills missing hours and days with count 0 per sensor
    :param data: dataset of log_sensor table from mock_up_db database
    :return clean_op_full: data with added missing hours
    """
    # group sensor data by hour and occupation. Count movements per sensor for every hour.
    clean_op = data.groupby(
        ['occupation_period_id', data.sensor_timestamp.dt.weekday, data.sensor_timestamp.dt.hour, data.sensor]).count()
    clean_op_reset = clean_op.rename_axis(["occupation_period", "weekday", 'hour', 'sensor']).reset_index().iloc[:, 0:5]
    clean_op_reset = clean_op_reset.rename(columns={'log_sensor_id': 'count'})

    # empty dataframe
    clean_op_full = pd.DataFrame(columns=['occupation_period', 'weekday', 'hour', 'sensor', 'count'])

    # add missing hours with count 0
    for i in tqdm(clean_op_reset.occupation_period.unique()):

        df = clean_op_reset[clean_op_reset['occupation_period'] == i]
        df = df[df['sensor'].isin(sensors)]

        for s in list(df.sensor.unique()):

            te = df[df['sensor'] == s]
            days = df.weekday.unique()

            for d in np.arange(0, 7):
                if d in days:
                    ta = te[te['weekday'] == d]
                    for h in np.arange(0, 24):
                        if h in ta.hour.unique():
                            th = ta[ta['hour'] == h]
                            val = int(th['count'])

                            clean_op_full = clean_op_full.append(
                                {'occupation_period': i, 'weekday': d, 'hour': h, 'sensor': s, 'count': val},
                                ignore_index=True)

                        else:
                            clean_op_full = clean_op_full.append(
                                {'occupation_period': i, 'weekday': d, 'hour': h, 'sensor': s, 'count': 0},
                                ignore_index=True)
                else:
                    clean_op_full = clean_op_full.append(
                        [{'occupation_period': i, 'weekday': d, 'hour': h, 'sensor': s, 'count': 0} for h in
                         range(0, 25)], ignore_index=True)

    return clean_op_full


# 'stringify' weekday column into readable weekdays
def dayNameFromWeekday(weekday):
    """
    Can be used to convert integer Weekday from pandas.datetime.weekday into strings (german)
    Monday = 0, Sunday = 7
    """
    days = ["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."]
    return days[weekday]


def sim_move(data, sensor1, sensor2):
    """
    Returns Events in which both 'Drehwand' and 'Drehschrank' have been moved and timedelta between them
    :param data: dataframe containing log_sensor table data
    :return sim_filtered: dataframe containing events of simultanous movement 
    """
    # reduce sensor data to 'Drehwand' & 'Drehschrank'
    data = data[data['sensor'].isin([sensor1, sensor2])]

    # order by timestamp
    data = data.sort_values('sensor_timestamp')
    data = data.reset_index()

    # create empty list
    sim_list = []

    # search for simultanous movement
    for i in tqdm(np.arange(len(data))):
        first = data.iloc[i]
        try:
            second = data.iloc[i + 1]
        except:
            pass

        # test if first and diffrent sensors are not the same
        if first['sensor'] != second['sensor']:
            duration = second['sensor_timestamp'] - first['sensor_timestamp']

            sim_list.append(
                [first['occupation_period_id'], first['log_sensor_id'], first['sensor'], second['log_sensor_id'],
                 second['sensor'], first['sensor_timestamp'], duration])

    sim = pd.DataFrame(sim_list, columns=['occupation_period_id', 'log_sensor1', 'sensor1', 'log_sensor2', 'sensor2',
                                          'time_of_event', 'time_diff'])

    return sim


def filter_sim(timedelta, data):
    """        
    Filters sim dataframe by timedelta
    :param data: (int) amount of seconds that can be between two movements and still be considered as a simultanous movement
    :retrun sim_filtered: filtered dataframe
    """
    sim = sim_move(data, 'Drehwand', 'Drehschrank')
    # filter by defined allowed timedelta
    sim_filtered = sim[sim['time_diff'].dt.total_seconds() <= timedelta]

    return sim_filtered


def sim_move_info(timedelta, data):
    @interact(timedelta=60)
    def _sim_move_info(timedelta):
        """
        Get information about distributions and percentages of sim_move dataframe
        """

        sim_filtered = filter_sim(timedelta, data)

        percent_sim = len(sim_filtered) / (len(data[data['sensor'].isin(['Drehwand', 'Drehschrank'])]) / 100)

        print('Verwendetes Zeitdelta: {} Sekunden \n'.format(timedelta))
        print('{0:30}{1:1}'.format('Variable', 'Test'))
        print('---------------------------------------')
        print('{0:30}{1:1}'.format('Simultane Bewegungen', len(sim_filtered)))
        print('{0:30}{1:1}'.format('Bewegungen Drehwand', len(data[data.sensor == 'Drehwand'])))
        print('{0:30}{1:1}'.format('Bewegungen Drehschrank', len(data[data.sensor == 'Drehschrank'])))
        print('{0:30}{1:1} %'.format('Prozent simultan bewegt', np.round(percent_sim, 2)))
        print('{0:30}{1:1} %'.format('Prozent einzeln bewegt', np.round(100 - percent_sim, 2)))


def sim_move_hour(data):
    """
    Takes data with n movement per hour and fills missing hours with value 0
    :param data: dataset containing log_sensor events
    :return event_hour: data filled with missing hours.
    """
    event_hour = pd.DataFrame(data.groupby(data.time_of_event.dt.hour)['log_sensor1'].count())
    event_hour = event_hour.reset_index()

    # add missing hours in dataframe for a better visualization
    for i in np.arange(0, 24):
        if i in list(event_hour['time_of_event']):
            pass
        else:
            event_hour.loc[len(event_hour)] = [i, 0]

    return event_hour

# todo add functionality for norm parameter
def sim_move_age_situation(data, person, norm=None):
    """
    Merges data about movement to plot amount of movement per age group and per living situation
    
    :param data: dataset containing movement data from filter_sim()-function
    :param person: person table dataset from mock_up_db database
    :param norm: DataFrame consisting available time per occupation_period
    
    :return data_age, data_status: two datasets containing infos about age group and living situation
    """
    # DataFrame containing the number of people per occupation_period
    status = pd.DataFrame(person.groupby('occupation_period_id')['person_id'].count()).rename(
        columns={'person_id': 'n_person'}).reset_index()

    # data for #1 plot
    data_age = pd.merge(data, person, on='occupation_period_id', how='left').iloc[:, :9]
    data_age = data_age.groupby('log_sensor1').first()
    data_age = data_age.groupby('age').count().iloc[:, :1].rename(
        columns={'occupation_period_id': 'count'}).reset_index()
    

    # add missing agecategorie if necessary
    """if '45-60' not in data_age.age:
        data_age.loc[len(data_age)] = ['45-60', 0]"""

    data_age = data_age.sort_values('age')

    # data for #2plot
    data_status = pd.merge(data, status, on='occupation_period_id', how='left')
    data_status = data_status.groupby('n_person').count().iloc[:, :1].reset_index()        

    return data_age, data_status


def time_interaction_table():
    """
        Takes data and returns the hourly number of interactions per hour for all sensors and each rotary element
        :return df: a dataset containing infos about the number of hourly interactions and if these hours
        where during the experiment
        """

    # Get table
    dba = DBAccess()
    table = dba.select('log_sensor_table')
    table = table.drop(
        ['log_sensor_id', 'sensor_subtype', 'sensor_state', 'sensor_numeric1', 'sensor_numeric2', 'sensor_diff',
         'sensor_duration', 'room', 'flag'], axis=1)

    # Get not occupied
    not_occupied = pd.read_csv('../database/not_occupied.csv', sep=',')

    # Count interactions for every hour
    interaction_count = table['sensor'].groupby(
        pd.to_datetime(table['sensor_timestamp'].dt.strftime("%Y/%m/%d %H"))).count().reset_index()
    interaction_count[['total', 'timestamp']] = interaction_count[['sensor', 'sensor_timestamp']]

    # Count "Drehwand" for every hour
    Drehwand = table[['sensor', 'sensor_timestamp']].loc[table['sensor'] == 'Drehwand']
    Drehwand[['Drehwand', 'timestamp']] = Drehwand[['sensor', 'sensor_timestamp']]
    Drehwand_count = Drehwand['Drehwand'].groupby(
        pd.to_datetime(Drehwand['timestamp'].dt.strftime("%Y/%m/%d %H"))).count().reset_index()

    # Count "Drehschrank" for every hour
    Drehschrank = table[['sensor', 'sensor_timestamp']].loc[table['sensor'] == 'Drehschrank']
    Drehschrank[['Drehschrank', 'timestamp']] = Drehschrank[['sensor', 'sensor_timestamp']]
    Drehschrank_count = Drehschrank['Drehschrank'].groupby(
        pd.to_datetime(Drehschrank['timestamp'].dt.strftime("%Y/%m/%d %H"))).count().reset_index()

    # Count "LampeDrehwand" for every hour
    LampeDrehwand = table[['sensor', 'sensor_timestamp']].loc[table['sensor'] == 'LampeDrehwand']
    LampeDrehwand[['LampeDrehwand', 'timestamp']] = LampeDrehwand[['sensor', 'sensor_timestamp']]
    LampeDrehwand_count = LampeDrehwand['LampeDrehwand'].groupby(
        pd.to_datetime(LampeDrehwand['timestamp'].dt.strftime("%Y/%m/%d %H"))).count().reset_index()

    # Count "LampeAussenwand" for every hour
    LampeAussenwand = table[['sensor', 'sensor_timestamp']].loc[table['sensor'] == 'LampeAussenwand']
    LampeAussenwand[['LampeAussenwand', 'timestamp']] = LampeAussenwand[['sensor', 'sensor_timestamp']]
    LampeAussenwand_count = LampeAussenwand['LampeAussenwand'].groupby(
        pd.to_datetime(LampeAussenwand['timestamp'].dt.strftime("%Y/%m/%d %H"))).count().reset_index()

    # Create dataframe
    start_date = pd.to_datetime(table.iloc[0, 3].strftime("%Y/%m/%d %H"))  # Get date from oldest timestamp in db
    end_date = pd.to_datetime(table.iloc[-1, 3].strftime("%Y/%m/%d %H"))  # Get date from newest timestamp in db
    date = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame({'timestamp': date, 'during_experiment': True})

    df['during_experiment'] = np.where(
        (df['timestamp'].dt.strftime("%H") >= '10') & (df['timestamp'].dt.strftime("%H") < '14') & (
                df['timestamp'].dt.strftime("%a") == 'Mon'), False, df['during_experiment'])
    for i in range(len(not_occupied)):
        df['during_experiment'] = np.where((df['timestamp'] >= pd.Timestamp(not_occupied.iloc[i]['empty_from'])) & (
                df['timestamp'] < pd.Timestamp(not_occupied.iloc[i]['empty_to'])), False, df['during_experiment'])

    df = df.merge(interaction_count[['total', 'timestamp']], on='timestamp', how='outer')
    df = df.merge(Drehwand_count[['Drehwand', 'timestamp']], on='timestamp', how='outer')
    df = df.merge(Drehschrank_count[['Drehschrank', 'timestamp']], on='timestamp', how='outer')
    df = df.merge(LampeDrehwand_count[['LampeDrehwand', 'timestamp']], on='timestamp', how='outer')
    df = df.merge(LampeAussenwand_count[['LampeAussenwand', 'timestamp']], on='timestamp', how='outer')

    # Fill empty hours with 0
    df = df.fillna(0)

    return df


def pairwise(iterable):
    """
    pairs every other two elements in an iterable
    """
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def get_times(query, sensor, move=False, turn=False):
    """
    Creates DataFrame which contains the usable time per occupation_period_id and sensor in query data.
    Query needs to contain data from the log_sensor table
    
    :param query: Query-Key for the select function
    :param move: (boolean) if True, uses data from malfunction.csv with usable == 1
    
    :return df_times: DataFrame containing usable time per occupation_period_id and sensor
    """
    # get data
    dba = DBAccess()
    df = dba.select(query)
    
    # filter data if turn
    df = df[df['sensor'].isin(sensor)]
    
    # get unique sensors
    sensors = df.sensor.unique()
    
    # append nones
    df = append_nones_complete(df, move=move)
    
    # initalize DataFrame
    df_timecount = pd.DataFrame(columns=['occupation_period_id',
                                         'sensor',
                                         'avail_min'])
    
    # loop through sensors and append
    for s in sensors:
        df_s = df[df['sensor']==s]
        df_s = df_s.iloc[1:-1,:].reset_index(drop=True) # trim first and last row (nones)
        
        # get list of indexes with rows which have 'log_sensor_id' == NaN
        index_cut = df_s[df_s['log_sensor_id'].isna()].index.tolist() # get index of nones
        
        for ind_p, ind_a in pairwise(index_cut):  # loops through index list and saves a row in df_timecount
    
            # get next row
            row = df_s.iloc[ind_p+1,:]

            # get occupation_period
            op = row['occupation_period_id']

            # get sensor
            s = row['sensor']
            
            # get available time as timedelta
            diff = (df_s.iloc[ind_a,:]['sensor_timestamp']- df_s.iloc[ind_p,:]['sensor_timestamp'])

            # append row to df
            df_timecount = df_timecount.append({'occupation_period_id':op,
                                                'sensor':s,
                                                'avail_min':diff}, ignore_index=True)
    
    # group by occupation_period_id
    df_timecount = df_timecount.groupby(by=['occupation_period_id','sensor']).sum().reset_index().sort_values(by=['sensor','occupation_period_id'])
            
    return df_timecount


def get_minutes(query, sensor, move=False, turn=False):
    """
    Creates DataFrame which contains the usable minutes per occupation_period_id, date and sensor in query data.
    Query needs to contain data from the log_sensor table
    
    :param query: Query-Key for the select function
    :param move: (boolean) if True, uses data from malfunction.csv with usable == 1
    
    :return df_times: DataFrame containing usable time per occupation_period_id and sensor
    """
    # get data
    dba = DBAccess()
    df = dba.select(query)
    
    
    df = df[df['sensor'].isin(sensor)]
    sensors = df.sensor.unique()

    df['date'] = df['sensor_timestamp'].astype('datetime64[D]')

    # append nones
    df = append_nones_complete(df, move=True)    
    
    # initalize DataFrame
    df_timecount = pd.DataFrame(columns=['occupation_period_id',
                                         'day',
                                         'sensor',
                                         'avail_min'])

    for s in sensors:
            df_s = df[df['sensor']==s]
            df_s = df_s.iloc[1:-1,:].reset_index(drop=True) # trim first and last row (nones)

            # get list of indexes with rows which have 'log_sensor_id' == NaN
            index_cut = df_s[df_s['log_sensor_id'].isna()].index.tolist() # get index of nones

            for ind_p, ind_a in pairwise(index_cut):  # loops through index list and saves a row in df_timecount

                # get next row
                row = df_s.iloc[ind_p+1,:]

                # get occupation_period
                op = row['occupation_period_id']

                # get sensor
                s = row['sensor']

                # get available time as timedelta
                date_p = df_s.iloc[ind_p,11].date()
                td_p = (datetime.datetime.combine((date_p + datetime.timedelta(days=1)), datetime.datetime.min.time()) - df_s.iloc[ind_p,11].to_pydatetime())
                m_p = datetime.timedelta.total_seconds(td_p)//60 # get minutes

                date_a = df_s.iloc[ind_a,11].date()
                td_a = (df_s.iloc[ind_a,11].to_pydatetime() - datetime.datetime.combine(date_a, datetime.datetime.min.time()))
                m_a = datetime.timedelta.total_seconds(td_a)//60 # get minutes

                if date_p == date_a:
                    dates = [date_p]
                    avail_time = [(df_s.iloc[ind_a, 11] - df_s.iloc[ind_p, 11]).total_seconds()//60]

                elif date_a > date_p:
                    days = [date_p + datetime.timedelta(days=i) for i in range(1,(date_a - date_p).days)]
                    avail_time = [m_p, *[24*60 for d in days], m_a]
                    dates = [date_p, *[d for d in days], date_a]

                # append row to df
                for i in range(len(dates)):
                    df_timecount = df_timecount.append({'occupation_period_id':op,
                                                    'day':dates[i],
                                                    'sensor':s,
                                                    'avail_min':avail_time[i]}, ignore_index=True)

    # group by occupation_period_id
    df_timecount = df_timecount.groupby(by=['occupation_period_id','sensor','day']).sum().reset_index().sort_values(by=['sensor','occupation_period_id'])
    df_timecount['weekday'] = df_timecount['day'].apply(lambda x: x.weekday())    
    
    return df_timecount

def filter_mal(usable):
    """
    Filter data out of log_sensor data where
    """
    
    mal = pd.read_csv('../database/Malfunction.csv') # table with malfunction timeframes

    dba = DBAccess()
    df = dba.select('log_sensor_table_flag_3_4')
    
    if usable == 1:
        for row in mal[mal['usable'] != 1].iterrows():
            row = row[1]
            df.drop(df[(df['sensor_timestamp'] >= row.start) & (df['sensor_timestamp'] <= row.end) & (df['sensor'] == row.sensorname)].index, inplace=True)
    else:
        pass
    return df

def add_min_max(sensor, data, weights):
    """
    Adds an artificial min and max value to `data` with extremly small `weights`
    to fill min and max of the weighted histogram if `data` is not empty
    In addition it generates the bin_edges
    
    :param sensor: str, either 'Drehwand', 'Drehschrank', 'LampeAussenwand' or 'LampeDrehwand'
    :param data: pd.Series
    :param weights: pd.Series, the weights of `data`
    
    :returns: `data` and `weights` modified and the number of bins required to have approx the binwidth defined in config.yml
    """
    ts = get_config()['turn_sensors']
    min_ = ts['Minimum']
    max_ = ts['Maximum_'+sensor]
    invis_weight = 0.
    
    data_S = pd.Series([min_, max_])
    weigths_S = pd.Series([invis_weight, invis_weight])
    data = data.append(data_S, ignore_index = True)
    weights = weights.append(weigths_S, ignore_index = True)
    
    edge_min, edge_max = data.min() - ts['binwidth'], data.max() + ts['binwidth']
    bin_edges = np.arange(min_ - ts['Toleranz'], max_ + ts['Toleranz'] + ts['binwidth'], ts['binwidth'])
    bin_edges = [n for n in bin_edges if (n > edge_min and n < edge_max)]
        
    return data, weights, bin_edges


def group_norm_category(df, sensor, category, situation, day_week='week', turn=True):
    """
    Group and normalize data to sensorm category and situation
    
    :param df: dataframe with log_sensor data
    :param sensor: sensor to display
    :param category: category to filter, 'age' or 'sex'
    :param situation: situation of inhabitants, 'single' or 'couple'
    :param day_week: if 'week' data gets grouped to week, if 'day' data gets grouped into weekdays
    :param turn: (bool) If True only sensors with type 'tunr' are considered
    
    :return df_grouped: grouped pandas DataFrame
    """
    dba = DBAccess()
    dict_dfs = {s:df[df['sensor']==s] for s in df.sensor.unique()}
    
    # data preparation
    person = dba.select('person')
    dict_dfs = merge_log_person(dict_dfs=dict_dfs,
                                df_person = living_situation_person(df_person=person),
                                category = category)

    # concat dict_dfs
    df_move_cat = pd.concat([dict_dfs[i] for i in df.sensor.unique()], ignore_index=True)

    if category == 'sex' and situation == 'couple':
        df_move_cat = df_move_cat[(df_move_cat['single_couple']=='couple') & (df_move_cat['single_sex_couple']==True)]
    
    # filter out couples/singles
    df_move_cat.drop(df_move_cat[df_move_cat['single_couple'] != situation].index, inplace=True)

    if day_week =='week':
        # data wrangling
        df_move_cat = df_move_cat.groupby(by=['occupation_period_id','sensor',category]) \
                                    .count() \
                                    .reset_index(drop=False) \
                                    .iloc[:,:4] \
                                    .rename(columns={'log_sensor_id':'count'})

        # normalize data
        df_avail = get_times('log_sensor_table_flag_3_4', sensor=sensor, move=True, turn=turn)

        df_move_cat = df_move_cat.merge(right=df_avail, on=['occupation_period_id','sensor'])

        df_move_cat['avail_min'] = df_move_cat['avail_min'].astype('timedelta64[m]')

        df_move_cat['norm'] = df_move_cat['count'] / df_move_cat['avail_min'] * 9955 # 9955 is a normal week in minutes

        df_grouped = df_move_cat.groupby(by=[category,'sensor']).agg({'occupation_period_id': 'count', 'norm': 'sum'}).reset_index()

        df_grouped['count_norm'] = df_grouped['norm'] / df_grouped['occupation_period_id']
        
    elif day_week == 'day':
        # data wrangling
        df_move_cat['day'] = df_move_cat['sensor_timestamp'].astype('datetime64[D]')

        df_move_cat = df_move_cat.groupby(by=['occupation_period_id','day','sensor',category]).count() \
                                    .reset_index(drop=False) \
                                    .rename(columns={'log_sensor_id':'count'}) \
                                    .iloc[:,:5]

        df_avail = get_minutes('log_sensor_table_flag_3_4', sensor=sensor, move=True, turn=turn)
        df_avail['day'] = df_avail['day'].astype('datetime64[D]')
        
        df_move_cat = df_move_cat.merge(right=df_avail, on=['occupation_period_id','sensor','day'])
        
        df_move_cat['norm'] = df_move_cat['count'] / df_move_cat['avail_min'] * 1440 # 1440 is a normal day in minutes
        
        df_grouped = df_move_cat.groupby(by=['weekday', category, 'sensor']).agg({'occupation_period_id': 'count', 'norm': 'sum'}).reset_index()

        df_grouped['count_norm'] = df_grouped['norm'] / df_grouped['occupation_period_id']

    return df_grouped












