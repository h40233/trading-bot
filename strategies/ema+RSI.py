import  pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("ema_RSI")
class ema_RSI(strategy):
    def __init__(self, name="ema_RSI", ema_length = 30, rsi_length = 9, lower = 30, upper = 70):
        super().__init__(name)
        self.ema_length = ema_length
        self.rsi_length = rsi_length
        self.lower = lower
        self.upper = upper

    def generate_signal(self, df):
        ema = ta.ema(df,self.ema_length)
        rsi = ta.rsi(df, self.rsi_length)
        df["ema"] = ema
        df["rsi"] = rsi
        df["signal"] = 0
        df.loc[ (df["close"] > df["ema"]) and (df["rsi"] < 30), "signal" ] = 1
        df.loc[ (df["close"] < df["ema"]) and (df["rsi"] > 70), "signal" ] = -1
        df["strategy_name"] = self.name
        return df