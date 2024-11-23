import pandas as pd
import sqlalchemy
from .base_fetcher import DataFetcher

class DatabaseFetcher(DataFetcher):
    """
    Fetches data from an SQL database.
    """
    def __init__(self, connection_string: str):
        """
        Initializes the DatabaseFetcher with a connection string.
        
        Parameters:
        ----------
        connection_string : str
            The SQLAlchemy connection string for the database.
        """
        self.connection_string = connection_string

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
            The query result.
        
        Raises:
        ------
        ValueError
            If the query fails or cannot fetch data.
        """
        try:
            engine = sqlalchemy.create_engine(self.connection_string)
            with engine.connect() as connection:
                return pd.read_sql(query, connection)
        except Exception as e:
            raise ValueError(f"Failed to execute query: {e}")
