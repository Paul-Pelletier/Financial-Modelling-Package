from abc import ABC, abstractmethod
from typing import Any, List

class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    """
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class Pipeline(ABC):
    """
    Abstract base class for a data processing pipeline.
    """
    
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
    
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        pass
