class strategy:
    def __init__(self, name = "strategy"):
        self.name = name
    
    def signal(self, df):
        """新策略必須有signal(df), 0 = none, 1 = buy, 2 = sell"""
        raise NotImplementedError("此策略還沒有signal(data)")
    
    def __str__(self):
        return self.name