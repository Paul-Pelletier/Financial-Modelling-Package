import unittest
import pandas as pd
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher
import sqlite3

class TestDatabaseFetcher(unittest.TestCase):
    def setUp(self):
        # Create an in-memory SQLite database for testing
        self.sqlite_connection_string = "sqlite:///:memory:"
        self.sqlite_connection = sqlite3.connect(":memory:")
        self.sqlite_connection.execute("CREATE TABLE test (id INT, value TEXT)")
        self.sqlite_connection.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b')")
        self.sqlite_connection.commit()

    def test_fetch_query_with_sqlalchemy(self):
        # Test with SQLAlchemy connection
        fetcher = DatabaseFetcher(self.sqlite_connection_string, use_sqlalchemy=True)

        # Fetch data and assert results
        query = "SELECT * FROM test"
        df = fetcher.fetch(query)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))  # Ensure 2 rows and 2 columns are fetched

        # Close the connection
        fetcher.close()

    def test_fetch_query_with_pyodbc(self):
        # Skip this test as pyodbc doesn't support SQLite's `sqlite:///:memory:` directly
        # Replace with a proper DSN or alternative database for real testing
        pass

    def tearDown(self):
        self.sqlite_connection.close()

if __name__ == "__main__":
    unittest.main()
