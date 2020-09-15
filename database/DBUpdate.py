# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:36:58 2020

@author: Lukas, Roman, Simon
"""

import io, yaml, requests, warnings
import pandas as pd
import numpy as np
from requests.auth import HTTPBasicAuth
from DBConnector import DBConnector
from datetime import datetime, timedelta

class DBUpdate(DBConnector):

    def __init__(self):
        super().__init__()
        
    def populate_sensor_tbl(self, drop=False):
        """
        Populates tbl with sensordata
        :param drop: drops content of table if set to True
        """
        # drop content
        if drop:
            self._drop_content('log_sensor')

        # gets the latest entry from db table "log_sensor"
        PostgreSQL_select_Query = "SELECT sensor_timestamp FROM log_sensor ORDER BY log_sensor_id DESC LIMIT 1"
        self._con.execute(PostgreSQL_select_Query)
        last_entry = self._con.fetchone()
        if last_entry == None:
            last_entry = '2019-08-19 00:00:00.000'
        else:
            last_entry = last_entry[0].strftime('%Y-%m-%d %H:%M:%S.%f')
        # gets new sensordata from Vacancy API as pandas DataFrame
        sensor_data = self._load_sensordata()
        new_sensor_data = sensor_data[sensor_data['Timestamp'] > last_entry]
        new_sensor_data = (self._set_flags(new_sensor_data)).replace(np.nan, '', regex=True)
        print("Load new sensordata...")
        percent_old = 0
        #create SQL statements
        for k in range(len(new_sensor_data)):
            flag = new_sensor_data.iloc[k]['Flag']
            sensor = new_sensor_data.iloc[k]['Sensorname']
            try:
                room = new_sensor_data.iloc[k]['Sensorname'][0]
            except:
                room = ""
            sensor_type = new_sensor_data.iloc[k]['Type']
            sensor_subtype = new_sensor_data.iloc[k]['Subtype']
            timestamp = "TIMESTAMP '{}'".format(new_sensor_data.iloc[k]['Timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f'))

            data = '''sensor_state'''

            # Switch or rotating element?(((dataframe['Timestamp']) >= pd.Timestamp(not_occupied.iloc[i]['empty_from'])) & ((dataframe['Timestamp']) < pd.Timestamp(not_occupied.iloc[i]['empty_to'])))
            if (new_sensor_data.iloc[k]['Value2']) == '':

                binary = True

                # Change switch values to binary ( 0, 1)
                if new_sensor_data.iloc[k]['Value1'] in {"OFF", "CLOSED"}:
                    sensor_state = 0


                elif new_sensor_data.iloc[k]['Value1'] in {"ON", "OPEN"}:
                    sensor_state = 1
                else:
                    data = '''sensor_numeric1'''
                    sensor_state = new_sensor_data.iloc[k]['Value1']

            else:
                data = '''sensor_numeric1, sensor_numeric2, sensor_diff, sensor_duration'''
                sensor_numeric1 = new_sensor_data.iloc[k]['Value1']
                sensor_numeric2 = new_sensor_data.iloc[k]['Value2']
                sensor_diff = new_sensor_data.iloc[k]['ValueDiff']
                sensor_duration = new_sensor_data.iloc[k]['Duration']
                binary = False

            columns = ''' sensor, occupation_period_id, sensor_type, sensor_subtype, room, {}, sensor_timestamp, flag'''.format(data)

            # Create SQL Insert Statment
            # check for occupation_period_id
            sensor_time = new_sensor_data.iloc[k]['Timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")
            query_timeslot = """SELECT occupation_period_id FROM occupation_period WHERE '{}' BETWEEN occupation_period.start AND occupation_period.end;""".format(sensor_time)
            self._con.execute(query_timeslot)
            timeslot = self._con.fetchone()
            if timeslot:
                occupation_period_id = timeslot[0]
            else:
                timeslot = None
                occupation_period_id = timeslot
            if binary:
                sql_statement = (
                    '''INSERT INTO log_sensor ({}) VALUES (NULLIF('{}',''), NULLIF('{}','None')::numeric, NULLIF('{}',''), NULLIF('{}',''), NULLIF('{}',''), NULLIF('{}','')::numeric, {}, {});'''.format(
                        columns, sensor, occupation_period_id, sensor_type, sensor_subtype, room, sensor_state, timestamp, flag))
            else:
                sql_statement = (
                    '''INSERT INTO log_sensor ({}) VALUES (NULLIF('{}',''), NULLIF('{}','None')::numeric, NULLIF('{}',''), NULLIF('{}',''), NULLIF('{}',''), NULLIF('{}','')::numeric, NULLIF('{}','')::numeric, NULLIF('{}','')::numeric, NULLIF('{}','')::numeric, {}, {});'''.format(
                        columns, sensor, occupation_period_id, sensor_type, sensor_subtype, room, sensor_numeric1, sensor_numeric2,
                        sensor_diff, sensor_duration, timestamp, flag))
            percent_new = int(100 * np.round(((k) / len(new_sensor_data)), 2))
            self._con.execute(sql_statement)

            #Commit data to the Database every 1 Percent
            if percent_new > percent_old:
                self._connection.commit()
                print(end="\r\r")
                print(percent_new, "%", end="")
                percent_old = percent_new

        print('\r\rSensor data is up to date')



    def populate_user_tbl(self, drop=False):
        """
        Populates tbl with inhabitants data
        :param drop: drops content of table if set to True
        """

        # drop content
        if drop:
            self._drop_content('person')

        # check if occupation_period is already filled
        PostgreSQL_select_Query = "SELECT * FROM person LIMIT 1"
        self._con.execute(PostgreSQL_select_Query)
        last_entry = self._con.fetchone()
        if last_entry == None:

            user_data = pd.read_csv(self._root_path+'database/inhabitants.csv', sep=',')
            for index, row in user_data.iterrows():
                populate = """
                INSERT INTO person VALUES (DEFAULT,{},'{}',NULLIF('{}','nan'),NULLIF('{}','nan'),NULLIF('{}','nan'),
                NULLIF('{}','nan'),NULLIF('{}','nan')::numeric, NULLIF('{}','nan')::numeric, NULLIF('{}','nan')::int);""".format(
                    row['occupation_period'],
                    str(row['age']),
                    str(row['sex']),
                    str(row['profession']),
                    row['employment_level'],
                    str(row['origin_living_situation']),
                    row['origin_housing_space'],
                    row['person'],
                    int(row['attendance']))
                self._con.execute(populate)
            self._connection.commit()
        else:
            pass

    def populate_occupation_period_tbl(self, drop=False):
        """
        Populates tbl with occupation_period data
        :param drop: drops content of table if set to True
        """
        # drop content
        if drop:
            self._drop_content('log_sensor')

        # check if occupation_period is already filled
        PostgreSQL_select_Query = "SELECT * FROM occupation_period LIMIT 1"
        self._con.execute(PostgreSQL_select_Query)
        last_entry = self._con.fetchone()
        if last_entry == None:

            # load and clean csv
            user_data = pd.read_csv(self._root_path+'database/inhabitants.csv', sep=',')
            user_data.fillna(value='NULL', inplace=True)
            user_data = user_data[['occupation_period', 'start', 'end']]

            # calculate number of days between start and end date (range)
            user_data['start_stamp'] = pd.to_datetime(user_data.start)
            user_data['end_stamp'] = pd.to_datetime(user_data.end)
            user_data['range'] = (user_data.end_stamp - user_data.start_stamp).dt.days

            # drop dublicates to prohibit a unique-constraint error for double pks
            user_data.drop_duplicates(inplace=True)

            # insert data row for row
            for index, row in user_data.iterrows():
                populate = """
                INSERT INTO occupation_period VALUES ({},{},'{}','{}');""".format(row['occupation_period'],
                                                                                      row['range'],
                                                                                      row['start'],
                                                                                      row['end'],
                                                                                      )
                self._con.execute(populate)

            # commit changes to db
            self._connection.commit()
        else:
            pass

    def _drop_content(self, table):
        """
        Drops content of a defined table
        :param table: name of the table which content will be dropped
        """
        drop_query = """DELETE FROM {};""".format(table)
        self._con.execute(drop_query)
        self._connection.commit()
        print('Content of "{}" table dropped'.format(table))
        
    def _load_sensordata(self):
        """
        Loads the sensordata of Vacancy API and returns as a pandas DataFrame
        :return returns df_sensor, a DataFrame containing the sensordata
        """
        df_sensor = None
        # load access credentials
        with open(self._root_path+"db_config.yml", 'r') as ymlfile:
            download = yaml.load(ymlfile, Loader=yaml.FullLoader)['download']
        try:
            response = requests.get(download['url'], auth=HTTPBasicAuth(download['user'], download['pw']))
            if response.status_code == 200:
                pass
            elif response.status_code == 500: # workaround because server sends a 500 response sometimes as the first answer
                response = requests.get(download['url'], auth=HTTPBasicAuth(download['user'], download['pw']))
            if response.status_code != 200:
                warnings.warn("Status Code {}: For more informations go to https://de.wikipedia.org/wiki/HTTP-Statuscode".format(response.status_code))
            else:
                # convert csv to pandas df
                data = response.content.decode('utf-8')
                df_sensor = pd.read_csv(io.StringIO(data), delimiter=';')
            
                # removes =" at the beginning and " at the end of the Timestamp
                df_sensor.iloc[:, -1] = df_sensor.iloc[:, -1].str.slice(start=2, stop=-1)
                df_sensor.iloc[:, -1] = pd.to_datetime(df_sensor.iloc[:, -1])  # convert dtype str to datetime64
        except Exception as e:
            print(e)
    
        return df_sensor
    
    
    def check_data(self, hours):
        """
        Checks when the latest download happened and initiates a new one if it happened before a certain time frame
        :param hours: time slot in hours
        """
    
        with open(self._root_path+"database/update_time.txt", "r") as fp:
            update_time = list(fp)[-1]
            update_time = update_time.strip()

        with open("../config.yml", 'r') as ymlfile:
            project_end = yaml.load(ymlfile, Loader=yaml.FullLoader)['general_info']['Projekt_Ende']

        timeformat = '%Y-%m-%d %H:%M:%S.%f'

        # check if project has ended already
        if datetime.today() >= datetime.strptime(project_end, timeformat):
            warnings.warn("Project has ended on "+project_end[:10]+". No new data will be pulled to prevent unwanted changes.")
    
        # check if data has been loaded in a given timescale
        elif (datetime.today() - timedelta(hours=hours)) <= datetime.strptime(update_time, timeformat):
            pass
    
        # load new data
        else:
            self.populate_sensor_tbl()
            with open(self._root_path+"database/update_time.txt", "w") as fp:
                fp.write(str(datetime.today()))  # saves current time in update_time.txt
    
    
    def _set_flags(self, dataframe):
        """Checks the dataframe for invalid values and sets the corresponding flags lower flag numbers overwrite higher values
        Flags: 1 = Missing data, 2 = No real movement, 3 = Malfunction expected, 4 = Values out of defined range, 5 = Time of event outside experiment
        :param dataframe: new data"""
    
        not_occupied = pd.read_csv(self._root_path+'database/not_occupied.csv', sep=',')
        malfunction = pd.read_csv(self._root_path+'database/Malfunction.csv', sep=',')
        inhabitants = pd.read_csv(self._root_path+'database/inhabitants.csv', sep=',')

        with open("../config.yml", 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        turn_sensors = config['turn_sensors']
        info = config['general_info']


        dataframe.insert(9, 'Flag', 0)
        conditions = []
        flags = []

        # Flag = 1
        no_entry = np.where(pd.isnull(dataframe))
        no_sensor_info = np.where((no_entry[1] < 5))
        no_timestamp = np.where((no_entry[1] == 8))
        df_id = list(set(no_entry[0][no_sensor_info]))
    
        if len(df_id) > 0:
            for i in range(len(df_id)):
                conditions.append((dataframe.id == (df_id[i] + 1)))
                flags.append(1)

        # Flag = 2
        conditions.append((dataframe['ValueDiff'].astype(float)) <= turn_sensors['Min_Drehung']),  # Minor movement
        conditions.append(((dataframe['Duration'].astype(float)) / (
                    dataframe['ValueDiff'].astype(float) + 0.01)) > turn_sensors['Min_Drehgeschwindigkeit'])  # Movement to slow
        flags.extend([2, 2])

        # Flag = 3
        for i in range(len(malfunction)):
            conditions.append(((dataframe['Timestamp']) >= pd.Timestamp(malfunction.iloc[i]['start'])) \
                              & ((dataframe['Timestamp']) < pd.Timestamp(malfunction.iloc[i]['end'])) \
                              & ((dataframe['Sensorname']) == malfunction.iloc[i]['sensorname']))
            flags.append(3)

        # Flag = 4
        check_df = dataframe.copy(deep=True)
        check_df = check_df.replace(['OFF', 'CLOSED', 'ON', 'OPEN', np.nan], [0, 0, 1, 1, 0])
        conditions.append((pd.to_numeric(check_df['Value1']) < (turn_sensors['Minimum'] - turn_sensors['Toleranz'])) &
                          (pd.to_numeric(check_df['Value2']) < (turn_sensors['Minimum'] - turn_sensors['Toleranz'])))
        conditions.append((check_df['Sensorname'] == 'LampeAussenwand') &
                          (pd.to_numeric(check_df['Value1']) >= (turn_sensors['Maximum_LampeAussenwand'] + turn_sensors['Toleranz'])) &
                          (pd.to_numeric(check_df['Value2']) >= (turn_sensors['Maximum_LampeAussenwand'] + turn_sensors['Toleranz'])))
        conditions.append((check_df['Sensorname'] == 'Drehschrank') &
                          (pd.to_numeric(check_df['Value1']) >= (turn_sensors['Maximum_Drehschrank'] + turn_sensors['Toleranz'])) &
                          (pd.to_numeric(check_df['Value2']) >= (turn_sensors['Maximum_Drehschrank'] + turn_sensors['Toleranz'])))
        conditions.append((check_df['Sensorname'] == 'Drehwand') &
                          (pd.to_numeric(check_df['Value1']) >= (turn_sensors['Maximum_Drehwand'] + turn_sensors['Toleranz'])) &
                          (pd.to_numeric(check_df['Value2']) >= (turn_sensors['Maximum_Drehwand'] + turn_sensors['Toleranz'])))
        conditions.append((check_df['Sensorname'] == 'LampeDrehwand') &
                          (pd.to_numeric(check_df['Value1']) >= (turn_sensors['Maximum_LampeDrehwand'] + turn_sensors['Toleranz'])) &
                          (pd.to_numeric(check_df['Value2']) >= (turn_sensors['Maximum_LampeDrehwand'] + turn_sensors['Toleranz'])))
        flags.extend([4, 4, 4, 4, 4])

        # Flag = 5
        inhabitants = inhabitants[inhabitants.attendance == 1]
        inhabitants = inhabitants.groupby(by='occupation_period').first().reset_index()
        inhabitants = inhabitants.astype({'start':'datetime64', 'end':'datetime64'})
        last_date = max(inhabitants.loc[:, 'end'])
        inhabitants.loc[:, 'end'] = inhabitants.loc[:, 'end'].shift(
            1)  # after the shift, end to start is the non_occupied time interval between each occupation
        first = True
        for i in range(len(inhabitants)):
            if first:
                first = False
                continue
            conditions.append((dataframe['Timestamp'] >= inhabitants.iloc[i]['end']) \
                              & (dataframe['Timestamp'] < inhabitants.iloc[i]['start']))
            flags.append(5)
            
        # handles time interval between last occupation and project end
        project_end = datetime.strptime(info['Projekt_Ende'], '%Y-%m-%d %H:%M:%S.%f')
        conditions.append((dataframe['Timestamp'] >= last_date) \
                          & (dataframe['Timestamp'] < project_end))
        flags.append(5)

        for i in range(len(not_occupied)):
            conditions.append(((dataframe['Timestamp']) >= pd.Timestamp(not_occupied.iloc[i]['empty_from'])) & ((dataframe['Timestamp']) < pd.Timestamp(not_occupied.iloc[i]['empty_to'])))
            flags.append(5)

        dataframe['Flag'] = np.select(conditions, flags)

    
        return dataframe
