# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 策略名稱：RSI + 布林通道 + MACD 反轉策略 (RSI_Bollinger_MACD)
# 來源：轉換自 TradingView Pine Script
# 邏輯：
#   1. 多單 (Long)：
#      - 價格 < 布林下軌 (超賣)
#      - RSI < 30 (超賣)
#      - MACD 柱狀圖 (Hist) > 前一根 (動能增強/跌勢趨緩)
#   2. 空單 (Short)：
#      - 價格 > 布林上軌 (超買)
#      - RSI > 70 (超買)
#      - MACD 柱狀圖 (Hist) < 前一根 (動能減弱/漲勢趨緩)
# -----------------------------------------------------------------------------------------

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("RSI_Bollinger_MACD")
class RSI_Bollinger_MACD(strategy):
    
    # [Function 說明]
    # 初始化參數，這裡定義的變數會被 app.py 自動偵測並顯示在「參數優化實驗室」
    def __init__(self, 
                 name="RSI_Bollinger_MACD",
                 rsi_period: int = 14,      # RSI 週期
                 boll_period: int = 20,     # 布林通道週期
                 boll_std: float = 2.0,     # 布林通道標準差
                 macd_fast: int = 12,       # MACD 快線
                 macd_slow: int = 26,       # MACD 慢線
                 macd_signal: int = 9,      # MACD 訊號線
                 rsi_oversold: int = 30,    # RSI 超賣區 (做多)
                 rsi_overbought: int = 70): # RSI 超買區 (做空)
        
        super().__init__(name)
        self.rsi_period = rsi_period
        self.boll_period = boll_period
        self.boll_std = boll_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, df):
        # 1. 計算 RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        # 2. 計算布林通道 (Bollinger Bands)
        # pandas_ta 回傳順序固定: [Lower, Mid, Upper, Bandwidth, Percent]
        bb = ta.bbands(df['close'], length=self.boll_period, std=self.boll_std)
        
        # 使用 iloc 避免欄位名稱命名問題
        if bb is not None and not bb.empty:
            df['bb_lower'] = bb.iloc[:, 0] # 下軌
            df['bb_upper'] = bb.iloc[:, 2] # 上軌
        else:
            df['bb_lower'] = df['close']
            df['bb_upper'] = df['close']

        # 3. 計算 MACD
        # pandas_ta 回傳順序通常為: [MACD Line, Histogram, Signal Line]
        # 但為了保險，我們確認一下 pandas_ta 的慣例。通常中間那欄(index 1)是 Hist (MACDh)
        macd_df = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        
        if macd_df is not None and not macd_df.empty:
            # 取得柱狀圖 (Histogram)
            df['hist'] = macd_df.iloc[:, 1]
        else:
            df['hist'] = 0

        # 4. 計算 MACD 動能條件
        # macdImproving: 當前 Hist > 前一根 Hist
        # macdWeakening: 當前 Hist < 前一根 Hist
        df['hist_prev'] = df['hist'].shift(1)
        condition_macd_improving = df['hist'] > df['hist_prev']
        condition_macd_weakening = df['hist'] < df['hist_prev']

        # 5. 初始化信號
        df["signal"] = 0
        
        # 6. 進場邏輯
        # 多單條件: 價格 < 下軌 AND RSI < 30 AND MACD改善
        long_cond = (df['close'] < df['bb_lower']) & \
                    (df['rsi'] < self.rsi_oversold) & \
                    (condition_macd_improving)
        
        # 空單條件: 價格 > 上軌 AND RSI > 70 AND MACD轉弱
        short_cond = (df['close'] > df['bb_upper']) & \
                     (df['rsi'] > self.rsi_overbought) & \
                     (condition_macd_weakening)

        # 7. 寫入信號
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        
        # 記錄策略名稱
        df["strategy_name"] = self.name
        
        # 清理暫存欄位 (可選)
        # df.drop(columns=['hist_prev'], inplace=True)
        
        return df