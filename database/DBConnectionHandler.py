# -*- coding: utf-8 -*-
import psycopg2
import yaml


class DBConnectionHandler(object):
    """
    This class is used to open the connection to the database and close it if necessary
    """
    _con = None
    _connection = None

    def __init__(self, root_path):
        # load access credentials
        with open(root_path+"db_config.yml", 'r') as ymlfile:
            sql = yaml.load(ymlfile, Loader=yaml.FullLoader)['sql']

        # check for connection
        try:

            connection = psycopg2.connect(user=sql['user'],
                                          password=sql['password'],
                                          host=sql['host'],
                                          port=sql['port'],
                                          database=sql['db'])
            cursor = connection.cursor()

            # Print PostgreSQL connection properties
            # print(connection.get_dsn_parameters(), "\n")

            # Print PostgreSQL version
            cursor.execute("SELECT version();")
            # record = cursor.fetchone()
            # print("You are connected to - ", record, "\n")

            DBConnectionHandler._con = cursor
            DBConnectionHandler._connection = connection

        except(Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)
        # TO DO: Add "all-round" exception

    @staticmethod
    def get_db_connection(root_path):
        # gets the connection to the database
        if DBConnectionHandler._con is None:
            DBConnectionHandler(root_path)
        return DBConnectionHandler._con, DBConnectionHandler._connection
    
    @staticmethod
    def close_db_connection():
        DBConnectionHandler._con.close()
        DBConnectionHandler._con = None
        DBConnectionHandler._connection = None


