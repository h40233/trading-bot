# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 策略名稱：SMA 均線乖離回歸策略 (SMA_MeanReversion)
# 策略邏輯：
#   利用長短均線的位置關係以及價格與長均線的乖離率 (Threshold) 來判斷進場點。
# -----------------------------------------------------------------------------------------

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("SMA_MeanReversion")
class SMA_MeanReversion(strategy):
    
    # [Function 說明]
    # 初始化參數
    # short_len: 短期 SMA 週期 (預設 13)
    # long_len: 長期 SMA 週期 (預設 59)
    # threshold: 乖離率閾值百分比 (預設 3.5%)
    def __init__(self, 
                 name="SMA_MeanReversion",
                 short_len: int = 13,
                 long_len: int = 59,
                 threshold: float = 3.5):
        
        super().__init__(name)
        self.short_len = short_len
        self.long_len = long_len
        self.threshold = threshold

    def generate_signal(self, df):
        # 1. 計算 SMA 指標
        df['short_sma'] = ta.sma(df['close'], length=self.short_len)
        df['long_sma'] = ta.sma(df['close'], length=self.long_len)
        
        # 2. 初始化信號
        df["signal"] = 0
        
        # 為了避免計算時沒資料，先把 NaN 填 0 (雖然後面邏輯判斷會自動變成 False，但這樣比較安全)
        df['short_sma'] = df['short_sma'].fillna(0)
        df['long_sma'] = df['long_sma'].fillna(0)

        # 3. 定義乖離率計算需要的數值 (閾值轉為小數點，例如 3.5% -> 0.035)
        # 根據你的公式：close * 閾值
        threshold_val = df['close'] * (self.threshold / 100.0)

        # 4. 做多邏輯 (Signal = 1)
        # when: close - long_SMA > close * 閾值
        cond1_long = (df['close'] - df['long_sma']) > threshold_val
        # and: short_SMA > close > long_SMA
        cond2_long = (df['short_sma'] > df['close']) & (df['close'] > df['long_sma'])
        
        long_signal = cond1_long & cond2_long

        # 5. 做空邏輯 (Signal = -1)
        # when: long_SMA - close > close * 閾值
        cond1_short = (df['long_sma'] - df['close']) > threshold_val
        # and: short_SMA < long_SMA
        cond2_short = df['short_sma'] < df['long_sma']
        # and: short_SMA < close < long_SMA
        cond3_short = (df['short_sma'] < df['close']) & (df['close'] < df['long_sma'])
        
        short_signal = cond1_short & cond2_short & cond3_short

        # 6. 寫入信號
        df.loc[long_signal, "signal"] = 1
        df.loc[short_signal, "signal"] = -1
        
        # 記錄策略名稱與指標值 (方便觀察)
        df["strategy_name"] = self.name
        
        return df