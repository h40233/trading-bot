# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 策略名稱：3EMA + MACD 趨勢策略 (ThreeEMA_MACD)
# 來源：轉換自 TradingView Pine Script
# 進場邏輯：
#   1. 多單 (Long)：
#      - 快線 (12) 上穿 慢線 (25)
#      - 收盤價 > 200 EMA (大趨勢向上)
#      - MACD 柱狀圖 > 0 (動能翻正)
#   2. 空單 (Short)：
#      - 快線 (12) 下穿 慢線 (25)
#      - 收盤價 < 200 EMA (大趨勢向下)
#      - MACD 柱狀圖 < 0 (動能翻負)
# -----------------------------------------------------------------------------------------

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("ThreeEMA_MACD")
class ThreeEMA_MACD(strategy):
    
    # [Function 說明]
    # 初始化參數，對應 Pine Script 的 input
    def __init__(self, 
                 name="ThreeEMA_MACD",
                 fast_len: int = 12,       # 快線長度
                 slow_len: int = 25,       # 慢線長度
                 trend_len: int = 200,     # 趨勢線長度
                 macd_fast: int = 26,      # MACD 快線
                 macd_slow: int = 100,     # MACD 慢線 (PineScript設定為100)
                 macd_signal: int = 9):    # MACD 訊號線
        
        super().__init__(name)
        self.fast_len = fast_len
        self.slow_len = slow_len
        self.trend_len = trend_len
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def generate_signal(self, df):
        # 1. 計算三條 EMA
        df['fast_ma'] = ta.ema(df['close'], length=self.fast_len)
        df['slow_ma'] = ta.ema(df['close'], length=self.slow_len)
        df['ma_200'] = ta.ema(df['close'], length=self.trend_len)
        
        # 2. 計算 MACD
        # pandas_ta 的 macd 回傳欄位通常包含: MACD, Histogram, Signal
        # 我們需要的是 Histogram (柱狀圖)
        macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        
        # 為了避免欄位名稱變動，我們用 iloc 取值 (通常 index 1 是 Histogram)
        if macd is not None and not macd.empty:
            df['hist'] = macd.iloc[:, 1]
        else:
            df['hist'] = 0

        # 3. 判斷 交叉 (Crossover / Crossunder)
        # 黃金交叉: 當前快線 > 當前慢線 且 上一根快線 <= 上一根慢線
        crossover = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        
        # 死亡交叉: 當前快線 < 當前慢線 且 上一根快線 >= 上一根慢線
        crossunder = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))

        # 4. 初始化信號
        df["signal"] = 0
        
        # 5. 進場邏輯 (結合趨勢與動能)
        # Long: 交叉向上 AND 價格在200均之上 AND MACD柱狀圖 > 0
        long_cond = crossover & (df['close'] > df['ma_200']) & (df['hist'] > 0)
        
        # Short: 交叉向下 AND 價格在200均之下 AND MACD柱狀圖 < 0
        short_cond = crossunder & (df['close'] < df['ma_200']) & (df['hist'] < 0)

        # 6. 寫入信號
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        
        # 記錄策略名稱
        df["strategy_name"] = self.name
        
        return df