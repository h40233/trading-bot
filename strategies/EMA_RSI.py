import  pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("EMA_RSI")
class ema_RSI(strategy):
    def __init__(self,
                name="EMA_RSI",
                ema_length:int|None = 30,
                rsi_length:int|None = 9,
                lower:int|None = 30,
                upper:int|None = 70):
        super().__init__(name)
        self.ema_length = ema_length
        self.rsi_length = rsi_length
        self.lower = lower
        self.upper = upper

    def generate_signal(self, df):
        ema = ta.ema(df['close'],self.ema_length)
        rsi = ta.rsi(df['close'], self.rsi_length)
        df["ema"] = ema
        df["rsi"] = rsi
        df["signal"] = 0
        df.loc[ (df["close"] > df["ema"]) & (df["rsi"] < self.lower), "signal" ] = 1
        df.loc[ (df["close"] < df["ema"]) & (df["rsi"] > self.upper), "signal" ] = -1
        df["strategy_name"] = self.name
        return df