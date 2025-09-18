from abc import ABC, abstractmethod
import pandas as pd

class strategy(ABC):
    """所有策略必須繼承這個父類別，保證方法一致"""

    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def generate_signal(self, df):
        """signal = 0: none, 1 : buy, -1 : sell"""
        raise NotImplementedError("此策略還沒有signal(data)")
    
    def __str__(self):
        return self.name