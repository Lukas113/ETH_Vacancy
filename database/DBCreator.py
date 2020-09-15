# -*- coding: utf-8 -*-
import yaml
from DBConnector import DBConnector
from DBUpdate import DBUpdate
from datetime import datetime

# Index:
# [] create_db() = creates database mock_up_db
# [x] clear_db() = clears complete mock_up_db database
# [x] create_tbl() = creates the empty tables for the data
# [] populate_sensor_tbl() = populates the sensor_tbl
# [x] populate_person_tbl() = populates the inhabitants_tbl with the inhabitants.csv
# [x] populate_period_tbl() = populates the occupation_period_tbl


class DBCreator(DBConnector):

    def __init__(self):
        super().__init__()

    def create_db(self):  # currently the database can't be created through the function because the connector requires
        # a database name for a connection
        """
        Creates the database
        """
        pass

        with open("../db_config.yml", 'r') as ymlfile:
            sql = yaml.load(ymlfile, Loader=yaml.FullLoader)['sql']

        create_db_query = '''CREATE DATABASE IF NOT EXISTS {}
        WITH
        OWNER = {}
        ENCODING = 'UTF8'
        CONNECTION LIMIT = -1;

        COMMENT ON DATABASE {}
        IS 'DB for Vacancy-Project';'''.format(sql['db'], sql['user'], sql['db'])

        self._con.execute(create_db_query)

        self._connection.commit()

        print("Database created successfully in PostgreSQL ")

    def _clear_db(self):
        """
        Clears tables in database
        """
        drop_db_query = '''DROP TABLE IF EXISTS log_sensor, occupation_period, person CASCADE;'''
        self._con.execute(drop_db_query)

    def _create_tbl(self):
        """
        Creates all necessary tables
        :param cursor: connection object from DBA() class
        """
        create_person_tbl = '''CREATE TABLE IF NOT EXISTS "person" (
              "person_id" SERIAL PRIMARY KEY,
              "occupation_period_id" int,
              "age" varchar(5),
              "sex" varchar,
              "profession" varchar(20),
              "employment_level" varchar,
              "origin_living_situation" varchar(10),
              "origin_housing_space" numeric,
              "name_eth" numeric,
              "attendance" int
            );'''
        self._con.execute(create_person_tbl)

        create_sensor_tbl = '''CREATE TABLE IF NOT EXISTS "log_sensor" (
              "log_sensor_id" SERIAL PRIMARY KEY,
              "occupation_period_id" int,
              "sensor" varchar(30),
              "sensor_type" varchar(10),
              "sensor_subtype" varchar(20),
              "room" varchar,
              "sensor_state" numeric,
              "sensor_numeric1" numeric,
              "sensor_numeric2" numeric,
              "sensor_diff" numeric,
              "sensor_duration" numeric,
              "sensor_timestamp" timestamp,
              "flag" numeric
                );'''
        self._con.execute(create_sensor_tbl)

        create_occupation_tbl = '''CREATE TABLE IF NOT EXISTS "occupation_period" (
              "occupation_period_id" int PRIMARY KEY,
              "range" int,
              "start" timestamp,
              "end" timestamp
            );'''
        self._con.execute(create_occupation_tbl)
        self._connection.commit()

    def _create_pk_fk(self):
        """
        Adds foreign keys to tables
        :param cursor: connection object from DBA() class
        """
        create_fk_query_1 = '''ALTER TABLE "log_sensor"
            ADD FOREIGN KEY ("occupation_period_id")
            REFERENCES "occupation_period" ("occupation_period_id");'''
        create_fk_query_2 = '''ALTER TABLE "person"
            ADD FOREIGN KEY ("occupation_period_id")
            REFERENCES "occupation_period" ("occupation_period_id");'''

        self._con.execute(create_fk_query_1)
        self._con.execute(create_fk_query_2)
        self._connection.commit()
        print('Relations have been added to the tables')

    def init_db(self, drop = False):
        """DB-setup controller"""
        dbc = DBCreator()
        dbc._clear_db()
        dbc._create_tbl()
        dbc._create_pk_fk()

        # occupation_period table must be loaded before log_sensor table
        dbu = DBUpdate()
        dbu.populate_occupation_period_tbl(drop=drop)
        dbu.populate_user_tbl(drop=drop)
        dbu.populate_sensor_tbl(drop=drop)
        with open(self._root_path+"database/update_time.txt", "w") as fp:
            fp.write(str(datetime.today()))  # saves current time in update_time.txt



if __name__ == "__main__":
    dbc = DBCreator()
    dbc.init_db()
