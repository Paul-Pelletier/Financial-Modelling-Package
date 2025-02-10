from .PipelinesAndSteps import PipelineStep
from typing import Any, List
from financial_modelling.data_acquisition.database_fetcher import DatabaseFetcher

class DataBaseFetcher(PipelineStep):
    """
    Concrete implementation of PipelineStep for data fetching.
    """
    def __init__(self, database_fetcher: DatabaseFetcher, query: str):
        self.database_fetcher = database_fetcher
        self.query = query

    def process(self, data: Any) -> Any:
        print("Fetching data from database...")
        fetched_data = self.database_fetcher.fetch(self.query)
        return {"raw_data": fetched_data}