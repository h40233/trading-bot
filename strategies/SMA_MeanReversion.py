# =========================================================================================
# 檔案名稱：strategies/SMA_MeanReversion.py
# 策略名稱：SMA 均線乖離回歸策略
# 策略類型：乖離回歸 (Mean Reversion based on Moving Average Deviation)
#
# 邏輯核心：
#   利用長短兩條 SMA (Simple Moving Average) 定義趨勢與乖離。
#   當價格偏離長期均線過遠 (超過 threshold)，且短期均線出現回歸跡象時進場。
# =========================================================================================

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("SMA_MeanReversion")
class SMA_MeanReversion(strategy):
    
    # [方法：初始化]
    # short_len: 短期 SMA (反應靈敏，用來確認短期轉折)。
    # long_len: 長期 SMA (反應遲鈍，視為價值中樞)。
    # threshold: 乖離率閾值 (例如 3.5 代表 3.5%)。
    def __init__(self, 
                 name="SMA_MeanReversion",
                 short_len: int = 13,
                 long_len: int = 59,
                 threshold: float = 3.5):
        
        super().__init__(name)
        self.short_len = short_len
        self.long_len = long_len
        self.threshold = threshold

    # [方法：產生訊號]
    def generate_signal(self, df):
        # 1. 計算長短 SMA
        df['short_sma'] = ta.sma(df['close'], length=self.short_len)
        df['long_sma'] = ta.sma(df['close'], length=self.long_len)
        
        # 2. 初始化信號
        df["signal"] = 0
        
        # [數據清洗] 填充 NaN
        # 避免計算初期因為均線尚未產生數值而導致邏輯判斷出錯。
        df['short_sma'] = df['short_sma'].fillna(0)
        df['long_sma'] = df['long_sma'].fillna(0)

        # 3. 計算乖離閾值數值
        # 這裡將百分比 (3.5) 轉為實際價格距離。
        # 例如: 價格 100 * 0.035 = 3.5。
        threshold_val = df['close'] * (self.threshold / 100.0)

        # 4. 做多邏輯 (超跌反彈)
        # 條件 A: 負乖離過大 -> (收盤價 - 長均線) 的距離 > 閾值
        # 注意：原程式碼邏輯 `(df['close'] - df['long_sma']) > threshold_val` 看起來像是「正乖離」？
        # 如果是做多，通常是價格遠低於均線。讓我們檢查下面的 cond2。
        # cond2: short > close > long -> 這代表價格在長均線之上？
        # [修正建議] 根據常見的回歸邏輯：
        #   做多應該是: 價格 < 長均線 (跌深) 且 乖離 > 閾值。
        #   但此處程式碼保留原始邏輯不更動，僅做註解解析。
        
        # 原始邏輯解讀：
        # cond1: 收盤價比長均線高出很多 (正乖離大)。
        # cond2: 短均線 > 收盤價 > 長均線 (短期回調但還在長多趨勢中)。
        # 結論：這看起來像是一個「多頭回調買入」策略，而非單純的跌深反彈。
        cond1_long = (df['close'] - df['long_sma']) > threshold_val
        cond2_long = (df['short_sma'] > df['close']) & (df['close'] > df['long_sma'])
        
        long_signal = cond1_long & cond2_long

        # 5. 做空邏輯
        # 條件 A: 長均線比收盤價高出很多 (負乖離大)。
        # 條件 B: 短均線 < 長均線 (空頭排列)。
        # 條件 C: 短均線 < 收盤價 < 長均線 (空頭反彈但受壓)。
        cond1_short = (df['long_sma'] - df['close']) > threshold_val
        cond2_short = df['short_sma'] < df['long_sma']
        cond3_short = (df['short_sma'] < df['close']) & (df['close'] < df['long_sma'])
        
        short_signal = cond1_short & cond2_short & cond3_short

        # 6. 寫入信號
        df.loc[long_signal, "signal"] = 1
        df.loc[short_signal, "signal"] = -1
        
        df["strategy_name"] = self.name
        
        return df

# ====== 備註區 ======
# 註1: 邏輯驗證
#      此策略的 `cond1_long` 使用 `close - long_sma > threshold`，這表示價格要在均線「上方」很遠。
#      結合 `cond2_long` (short > close)，這捕捉的是「強勢股回檔」的瞬間——
#      大趨勢向上 (價>長)，但短期有修正 (短>價)，且修正幅度還沒跌破長均線。