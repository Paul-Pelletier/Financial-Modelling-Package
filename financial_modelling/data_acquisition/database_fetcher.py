import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import time

class DatabaseFetcher:
    """
    Fetches data from an SQL database.
    
    Supports both pyodbc and sqlalchemy for fetching data.
    Tracks and displays connection time for performance monitoring.
    """
    def __init__(self, connection_string: str, use_sqlalchemy: bool = False):
        """
        Initializes the DatabaseFetcher and times the connection setup.

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
        self.connection_time = None  # Store connection time for performance tracking

        if self.use_sqlalchemy:
            try:
                start_time = time.time()
                self.engine = create_engine(self.connection_string)
                end_time = time.time()
                self.connection_time = end_time - start_time
                print(f"SQLAlchemy engine created in {self.connection_time * 1000:.2f} ms")
            except Exception as e:
                raise ValueError(f"Failed to create SQLAlchemy engine: {e}")
        else:
            try:
                start_time = time.time()
                self.conn = pyodbc.connect(self.connection_string)
                end_time = time.time()
                self.connection_time = end_time - start_time
                print(f"pyodbc connection established in {self.connection_time * 1000:.2f} ms")
            except Exception as e:
                raise ValueError(f"Failed to connect using pyodbc: {e}")

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
                cursor = self.conn.cursor()
                start_time = time.time()
                df = pd.read_sql(query, self.conn)
                end_time = time.time()
                print(f"Query executed in {end_time - start_time:.4f} seconds")
                return df
            except Exception as e:
                raise ValueError(f"Failed to execute query using pyodbc: {e}")

    def close(self):
        """Closes the database connection."""
        if not self.use_sqlalchemy:
            try:
                self.conn.close()
                print("pyodbc connection closed.")
            except Exception as e:
                print(f"Error closing pyodbc connection: {e}")
