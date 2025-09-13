import  talib as ta
from strategy import strategy
class ema_RSI(strategy):
    def __init__(self, name="ema+RSI", ema_length = 30, rsi_length = 9, lower = 30, upper = 70):
        super().__init__(name)
        self.ema_length = ema_length
        self.rsi_length = rsi_length
        self.lower = lower
        self.upper = upper

    def signal(self, df):
        ema = ta.EMA(df, self.ema_length)
        rsi = ta.RSI(df, self.rsi_length)

