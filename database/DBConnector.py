from DBConnectionHandler import DBConnectionHandler

class DBConnector(object):  # superclass

    def __init__(self, root_path = '../'):
        self._con, self._connection = DBConnectionHandler.get_db_connection(root_path)
        self._root_path = root_path

    def close_connection(self):
        """
        Closes the connection to the database before destructing it
        """
        DBConnectionHandler.close_db_connection()


if __name__ == "__main__":
    DBConnector()


