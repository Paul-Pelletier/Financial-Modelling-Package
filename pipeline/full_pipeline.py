#from .pipeline.export_distinct_unixtimequotedate_pipeline import UniqueQuoteTimeFetcher
#from .pipeline.fetch_and_split_to_multiple_quote_dates_pipeline import FetchAndSplitToMultipleQuoteDates
#from .pipeline.calibrate_multiple_dates_pipeline import CalibrateMultipleDatesPipeline
from ..utils.utils import get_file_names

fetcher_and_splitter = FetchAndSplitToMultipleQuoteDates()
#fetcher_and_splitter.run()

source_folder_path = fetcher_and_splitter.folder_path
files_list = get_file_names(source_folder_path)

pass