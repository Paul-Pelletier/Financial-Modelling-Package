import unittest
import pandas as pd
from data_acquisition.database_fetcher import DatabaseFetcher
import sqlite3

class TestDatabaseFetcher(unittest.TestCase):
    def setUp(self):
        # Create an in-memory SQLite database for testing
        self.connection_string = "sqlite://"
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE test (id INT, value TEXT)")
        self.connection.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b')")
        self.connection.commit()

    def test_fetch_query(self):
        fetcher = DatabaseFetcher("sqlite:///:memory:")
        df = fetcher.fetch("SELECT * FROM test")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))

    def tearDown(self):
        self.connection.close()

if __name__ == "__main__":
    unittest.main()
