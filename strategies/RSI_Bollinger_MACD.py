# =========================================================================================
# 檔案名稱：strategies/RSI_Bollinger_MACD.py
# 策略名稱：RSI + 布林通道 + MACD 綜合策略
# 策略類型：動能確認的反轉策略 (Momentum-Confirmed Reversal)
#
# 邏輯核心：
#   除了要求價格極端 (布林) 與超買超賣 (RSI) 外，
#   額外加入 MACD 柱狀圖的變化作為「煞車訊號」。
#   - 做多：需要 MACD 柱狀圖 > 前一根 (代表下跌動能正在衰退)。
#   - 做空：需要 MACD 柱狀圖 < 前一根 (代表上漲動能正在衰退)。
# =========================================================================================

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("RSI_Bollinger_MACD")
class RSI_Bollinger_MACD(strategy):
    
    # [方法：初始化]
    # 這裡定義了多個參數，這些參數都會自動顯示在 Streamlit 的優化器介面中。
    def __init__(self, 
                 name="RSI_Bollinger_MACD",
                 rsi_period: int = 14,      # RSI 週期
                 boll_period: int = 20,     # 布林通道週期
                 boll_std: float = 2.0,     # 布林通道標準差
                 macd_fast: int = 12,       # MACD 快線
                 macd_slow: int = 26,       # MACD 慢線
                 macd_signal: int = 9,      # MACD 訊號線
                 rsi_oversold: int = 30,    # RSI 超賣區
                 rsi_overbought: int = 70): # RSI 超買區
        
        super().__init__(name)
        # 將參數存入實例變數
        self.rsi_period = rsi_period
        self.boll_period = boll_period
        self.boll_std = boll_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    # [方法：產生訊號]
    def generate_signal(self, df):
        # 1. 計算 RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        # 2. 計算布林通道
        bb = ta.bbands(df['close'], length=self.boll_period, std=self.boll_std)
        
        # [防呆] 確保 bb 有計算出來，並依位置取下軌(0)與上軌(2)。
        if bb is not None and not bb.empty:
            df['bb_lower'] = bb.iloc[:, 0]
            df['bb_upper'] = bb.iloc[:, 2]
        else:
            df['bb_lower'] = df['close']
            df['bb_upper'] = df['close']

        # 3. 計算 MACD
        # pandas_ta.macd 回傳順序通常為: [MACD Line, Histogram, Signal Line]。
        macd_df = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        
        if macd_df is not None and not macd_df.empty:
            # 取第 1 欄位：Histogram (柱狀圖)。這代表 MACD 線與訊號線的距離。
            df['hist'] = macd_df.iloc[:, 1]
        else:
            df['hist'] = 0

        # 4. 計算 MACD 動能條件
        # 使用 .shift(1) 取得「上一根 K 棒」的柱狀圖數值。
        df['hist_prev'] = df['hist'].shift(1)
        
        # 動能改善 (Improving)：當前柱狀圖 > 上一根 (綠柱變長 or 紅柱變短)。
        condition_macd_improving = df['hist'] > df['hist_prev']
        # 動能轉弱 (Weakening)：當前柱狀圖 < 上一根 (紅柱變長 or 綠柱變短)。
        condition_macd_weakening = df['hist'] < df['hist_prev']

        # 5. 初始化信號
        df["signal"] = 0
        
        # 6. 進場邏輯 (三合一條件)
        # 多單: 價格破下軌 + RSI超賣 + MACD動能轉好 (接刀更安全)。
        long_cond = (df['close'] < df['bb_lower']) & \
                    (df['rsi'] < self.rsi_oversold) & \
                    (condition_macd_improving)
        
        # 空單: 價格破上軌 + RSI超買 + MACD動能轉弱 (摸頭更安全)。
        short_cond = (df['close'] > df['bb_upper']) & \
                     (df['rsi'] > self.rsi_overbought) & \
                     (condition_macd_weakening)

        # 7. 寫入信號
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        
        df["strategy_name"] = self.name
        
        return df

# ====== 備註區 ======
# 註1: MACD Histogram 的意義
#      MACD 柱狀圖其實就是 MACD 快慢線差值的「加速度」。
#      當柱狀圖開始縮短時，意味著原有的趨勢正在減速，這通常是反轉發生的第一個徵兆。