# =========================================================================================
# 檔案名稱：strategies/ThreeEMA_MACD.py
# 策略名稱：3EMA + MACD 趨勢策略
# 策略類型：趨勢跟隨 (Trend Following)
#
# 邏輯核心：
#   這是一個多重濾網策略：
#   1. 趨勢濾網：價格必須在 200 EMA 之上 (只做多) 或之下 (只做空)。
#   2. 訊號觸發：快慢線 EMA 交叉 (Golden Cross / Death Cross)。
#   3. 動能確認：MACD 柱狀圖必須配合方向 (多頭要 > 0，空頭要 < 0)。
# =========================================================================================

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("ThreeEMA_MACD")
class ThreeEMA_MACD(strategy):
    
    # [方法：初始化]
    # 定義了三條 EMA 與 MACD 的參數。
    def __init__(self, 
                 name="ThreeEMA_MACD",
                 fast_len: int = 12,       # 快線 EMA
                 slow_len: int = 25,       # 慢線 EMA
                 trend_len: int = 200,     # 趨勢線 EMA (判斷牛熊分界)
                 macd_fast: int = 26,      # MACD 參數
                 macd_slow: int = 100,     # MACD 慢線 (設定為 100 較為長線)
                 macd_signal: int = 9):    # MACD 訊號線
        
        super().__init__(name)
        self.fast_len = fast_len
        self.slow_len = slow_len
        self.trend_len = trend_len
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    # [方法：產生訊號]
    def generate_signal(self, df):
        # 1. 計算三條 EMA
        df['fast_ma'] = ta.ema(df['close'], length=self.fast_len)
        df['slow_ma'] = ta.ema(df['close'], length=self.slow_len)
        df['ma_200'] = ta.ema(df['close'], length=self.trend_len)
        
        # 2. 計算 MACD
        # 注意：這裡的 MACD 慢線參數設為 100，比標準的 26 慢很多，意在過濾短期雜訊。
        macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        
        # 取出 Histogram (柱狀圖)
        if macd is not None and not macd.empty:
            df['hist'] = macd.iloc[:, 1]
        else:
            df['hist'] = 0

        # 3. 判斷交叉 (Crossover / Crossunder)
        # [核心技巧] 偵測交叉點
        # 黃金交叉定義：
        #   (1) 當前 K 線：快線 > 慢線
        #   (2) 上一根 K 線：快線 <= 慢線 (使用 .shift(1) 取得)
        # 兩者同時成立，代表剛剛發生了「向上穿越」。
        crossover = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        
        # 死亡交叉定義：
        #   (1) 當前 K 線：快線 < 慢線
        #   (2) 上一根 K 線：快線 >= 慢線
        crossunder = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))

        # 4. 初始化信號
        df["signal"] = 0
        
        # 5. 進場邏輯 (多重濾網)
        # 做多：
        #   1. 發生黃金交叉 (短期轉強)。
        #   2. 收盤價 > 200 EMA (長期趨勢向上)。
        #   3. MACD 柱狀圖 > 0 (動能支持)。
        long_cond = crossover & (df['close'] > df['ma_200']) & (df['hist'] > 0)
        
        # 做空：
        #   1. 發生死亡交叉 (短期轉弱)。
        #   2. 收盤價 < 200 EMA (長期趨勢向下)。
        #   3. MACD 柱狀圖 < 0 (動能支持)。
        short_cond = crossunder & (df['close'] < df['ma_200']) & (df['hist'] < 0)

        # 6. 寫入信號
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        
        df["strategy_name"] = self.name
        
        return df

# ====== 備註區 ======
# 註1: 趨勢濾網的重要性
#      使用 `ma_200` 作為濾網是華爾街最古老的格言之一：「不要與趨勢作對」。
#      這能有效避免在空頭市場中因為短期的反彈而誤做多單（接刀），或者在多頭市場中亂空。