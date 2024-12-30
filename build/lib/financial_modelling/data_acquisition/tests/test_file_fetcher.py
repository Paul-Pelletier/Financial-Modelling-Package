import unittest
import pandas as pd
from financial_modelling.data_acquisition.file_fetcher import FileFetcher

class TestFileFetcher(unittest.TestCase):
    def test_fetch_csv(self):
        fetcher = FileFetcher()
        test_data = "test.csv"
        with open(test_data, "w") as f:
            f.write("col1,col2\n1,2\n3,4")
        df = fetcher.fetch(test_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))

if __name__ == "__main__":
    unittest.main()
