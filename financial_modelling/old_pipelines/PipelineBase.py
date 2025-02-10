from abc import ABC, abstractmethod

class PipelineBase(ABC):
    def __init__(self, fetcher, output_folder):
        self.fetcher = fetcher
        self.output_folder = output_folder

    @abstractmethod
    def fetch_data(self, **kwargs):
        """
        Fetch data using the fetcher. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def process_data(self, **kwargs):
        """
        Process the fetched data. Must be implemented by subclasses.
        """
        pass
    @abstractmethod

    def fit_model(self, **kwargs):
        """
        Fits the model to a parametric model
        """
        pass
    
    def plot_fitted_model(self, **kwargs):
        """
        Plots the fitted model and the train data
        """
    @abstractmethod
    def save_output(self, data, **kwargs):
        """
        Save the processed data to the output folder. Must be implemented by subclasses.
        """
        pass

    def run(self, **kwargs):
        """
        Execute the pipeline.
        """
        data = self.fetch_data(**kwargs)
        processed_data = self.process_data(data, **kwargs)
        self.fit_model(**kwargs)
        self.save_output(processed_data, **kwargs)
