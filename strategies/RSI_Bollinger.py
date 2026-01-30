# =========================================================================================
# 檔案名稱：strategies/RSI_Bollinger.py
# 策略名稱：RSI + 布林通道逆勢策略
# 策略類型：均值回歸 (Mean Reversion)
#
# 交易邏輯：
#   1. 做多 (Long)：價格跌破布林下軌 (超賣) 且 RSI 低於 30 -> 預期反彈。
#   2. 做空 (Short)：價格突破布林上軌 (超買) 且 RSI 高於 70 -> 預期回檔。
# =========================================================================================

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("RSI_Bollinger")
class RSI_Bollinger(strategy):
    
    # [方法：初始化]
    # 設定 RSI 與布林通道 (BBands) 的計算參數。
    def __init__(self, 
                 name="RSI_Bollinger",
                 rsi_length: int = 14,      # RSI 週期，標準為 14
                 bb_length: int = 20,       # 布林通道 SMA 週期，標準為 20
                 bb_std: float = 2.0,       # 標準差倍數，通常設為 2.0
                 rsi_oversold: int = 30,    # RSI 超賣區 (做多門檻)
                 rsi_overbought: int = 70): # RSI 超買區 (做空門檻)
        
        super().__init__(name)
        self.rsi_length = rsi_length
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    # [方法：產生訊號]
    def generate_signal(self, df):
        # 1. 計算 RSI 指標
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_length)
        
        # 2. 計算布林通道 (Bollinger Bands)
        # pandas_ta 的 bbands 函數會回傳一個 DataFrame，包含 5 個欄位：
        # [Lower, Mid, Upper, Bandwidth, Percent]
        bb = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
        
        # [修正] 使用 iloc 依據位置取值
        # 這是為了避免 pandas_ta 欄位名稱變動 (例如 "BBL_20_2.0" vs "BBL_20_2") 的問題。
        # 依位置取值 (Column Index) 是最穩健的做法。
        if bb is not None and not bb.empty:
            df['bb_lower'] = bb.iloc[:, 0] # 取第 0 欄 (Lower Band / 下軌)
            df['bb_upper'] = bb.iloc[:, 2] # 取第 2 欄 (Upper Band / 上軌)
        else:
            # 防呆：如果資料不足導致無法計算，將上下軌設為收盤價 (避免觸發任何訊號)。
            df['bb_lower'] = df['close']
            df['bb_upper'] = df['close']

        # 3. 初始化信號欄位
        df["signal"] = 0
        
        # 4. 定義進場邏輯
        # 做多條件：價格跌破下軌 (極端弱勢) AND RSI 進入超賣區 (確認動能低檔)。
        long_condition = (df['close'] < df['bb_lower']) & (df['rsi'] < self.rsi_oversold)
        
        # 做空條件：價格突破上軌 (極端強勢) AND RSI 進入超買區 (確認動能高檔)。
        short_condition = (df['close'] > df['bb_upper']) & (df['rsi'] > self.rsi_overbought)
        
        # 5. 寫入信號
        # 使用 Pandas 的向量化操作 .loc 進行批次賦值。
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1
        
        # 記錄策略名稱，方便回測結果分析。
        df["strategy_name"] = self.name
        
        return df

# ====== 備註區 ======
# 註1: 布林通道的特性
#      布林通道會隨波動率放大縮小。在劇烈單邊行情中，價格可能會沿著上軌一直漲 (Walk the band)。
#      這類純逆勢策略在強趨勢中容易連續虧損，建議搭配 ADX 或其他濾網來過濾盤整盤。