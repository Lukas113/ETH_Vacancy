# -*- coding: utf-8 -*-
from DBConnector import DBConnector
from DBUpdate import DBUpdate
import pandas as pd

class DBAccess(DBConnector):
    
    _dtypes = {'log_sensor_id':'int64',
               'occupation_period_id':'float64', #not int64 due to nullable records
               'sensor':'object',
               'sensor_type':'object',
               'sensor_subtype':'object',
               'room':'object',
               'sensor_state':'float64', #not int64 due to nullable records
               'sensor_numeric1':'float64',
               'sensor_numeric2':'float64',
               'sensor_diff':'float64',
               'sensor_duration':'float64',
               'sensor_timestamp':'datetime64',
               'flag':'int64',
               'range':'int64',
               'start':'datetime64',
               'end':'datetime64',
               'person_id':'int64',
               'age':'object',
               'sex':'object',
               'profession':'object',
               'employment_level':'object',
               'origin_living_situation':'object',
               'origin_housing_space':'float64',
               'name_eth':'float64',
               'attendance':'int64'}

    def __init__(self):
        super().__init__()
        self._select_queries = {}
        
        self._select_queries['rotary_elements'] = "SELECT * FROM log_sensor WHERE sensor_type = 'turn';"
        
        self._select_queries['person'] = "SELECT * FROM person"
        
        self._select_queries['sensor_data'] = """SELECT op.occupation_period_id,
                            log_sensor_id, sensor, sensor_type, sensor_subtype, sensor_numeric1, sensor_numeric2, sensor_diff, sensor_timestamp, room
                            FROM person pe INNER JOIN occupation_period op ON pe.occupation_period_id = op.occupation_period_id
                            INNER JOIN log_sensor ls ON op.occupation_period_id = ls.occupation_period_id
                            WHERE attendance = 1 AND flag = 0;"""
                            
        self._select_queries['sensor_data_dirty'] = """SELECT op.occupation_period_id,
                            log_sensor_id, sensor, sensor_type, sensor_subtype, sensor_numeric1, sensor_numeric2, sensor_diff, sensor_timestamp
                            FROM person pe INNER JOIN occupation_period op ON pe.occupation_period_id = op.occupation_period_id
                            INNER JOIN log_sensor ls ON op.occupation_period_id = ls.occupation_period_id
                            WHERE attendance = 1;"""
                            
        self._select_queries['sensor_person'] = """SELECT person_id, op.occupation_period_id, age, sex, profession,
                            log_sensor_id, sensor, sensor_type, sensor_subtype, room, sensor_numeric1, sensor_numeric2, sensor_diff, sensor_timestamp
                            FROM person pe INNER JOIN occupation_period op ON pe.occupation_period_id = op.occupation_period_id
                            INNER JOIN log_sensor ls ON op.occupation_period_id = ls.occupation_period_id
                            WHERE attendance = 1 AND flag = 0;"""
                            
        self._select_queries['turn_sensor_occupation'] = """SELECT op.occupation_period_id, log_sensor_id, sensor, room, 
                            sensor_numeric1, sensor_numeric2, sensor_diff, sensor_timestamp, flag
                            FROM occupation_period op INNER JOIN log_sensor ls ON op.occupation_period_id = ls.occupation_period_id
                            WHERE (flag = 0 or flag = 4) AND sensor_type = 'turn';"""
                            
        self._select_queries['occupation_period'] = """SELECT * FROM occupation_period"""

        self._select_queries['log_sensor_table'] = "SELECT * FROM log_sensor WHERE flag = 0 AND occupation_period_id >= 1 AND sensor_subtype not like 'LightPercent' AND sensor_subtype not like 'MoistPercent' AND sensor_subtype not like 'Temperature' "
        self._select_queries['log_sensor_table_flag_3_4'] = "SELECT * FROM log_sensor WHERE flag = 0 OR flag = 3 OR flag = 4 AND occupation_period_id >= 1 AND sensor_subtype not like 'LightPercent' AND sensor_subtype not like 'MoistPercent' AND sensor_subtype not like 'Temperature' "

        self._select_queries['person_table'] = "SELECT * FROM person"
        self._select_queries['sensor_data_total'] = "SELECT * FROM log_sensor;"
        self._select_queries['occupation_period_table'] = "SELECT * FROM occupation_period"

    def select(self, query_key):
        """
        Executes an sql select query to the connected database (postgres) and converts it to a pandas.DataFrame
        :param query_key: str containing an sql query_key to execute the according query
        """
        dbu = DBUpdate()
        dbu.check_data(1)
        self._con.execute(self._select_queries[query_key])
        data = self._con.fetchall()
        columns = [desc[0] for desc in self._con.description]
        
        #makes sure that every column has the correct dtype
        selected_dtypes = {k:DBAccess._dtypes[k] for k in DBAccess._dtypes if k in columns}
        df = pd.DataFrame(data, columns = columns)
        df = df.astype(selected_dtypes)
        return df
