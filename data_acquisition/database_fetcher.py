import pandas as pd
import pyodbc
from sqlalchemy import create_engine

class DatabaseFetcher:
    """
    Fetches data from an SQL database.
    
    Supports both pyodbc and sqlalchemy for fetching data.
    """
    def __init__(self, connection_string: str, use_sqlalchemy: bool = False):
        """
        Initializes the DatabaseFetcher.

        Parameters:
        ----------
        connection_string : str
            The connection string for the database.
            - For pyodbc: Use standard pyodbc format.
            - For sqlalchemy: Use sqlalchemy-compliant connection string.
        use_sqlalchemy : bool, optional
            If True, uses sqlalchemy for the connection. Otherwise, uses pyodbc.
        """
        self.connection_string = connection_string
        self.use_sqlalchemy = use_sqlalchemy

        if self.use_sqlalchemy:
            try:
                self.engine = create_engine(self.connection_string)
            except Exception as e:
                raise ValueError(f"Failed to create SQLAlchemy engine: {e}")

    def fetch(self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.

        Parameters:
        ----------
        query : str
            The SQL query to execute.

        Returns:
        -------
        pd.DataFrame
            The query result as a Pandas DataFrame.
        """
        if self.use_sqlalchemy:
            # Use SQLAlchemy for connection
            try:
                with self.engine.connect() as conn:
                    return pd.read_sql(query, conn)
            except Exception as e:
                raise ValueError(f"Failed to execute query using SQLAlchemy: {e}")
        else:
            # Use pyodbc for connection
            try:
                conn = pyodbc.connect(self.connection_string)
                df = pd.read_sql(query, conn)
                conn.close()
                return df
            except Exception as e:
                raise ValueError(f"Failed to execute query using pyodbc: {e}")
