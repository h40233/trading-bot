# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 策略名稱：RSI + 布林通道逆勢策略 (RSI_Bollinger)
# 策略邏輯：
#   1. 做多 (Long)：當價格跌破布林下軌 (超賣) 且 RSI 低於超賣線 (如 30) -> 預期反彈
#   2. 做空 (Short)：當價格突破布林上軌 (超買) 且 RSI 高於超買線 (如 70) -> 預期回調
# -----------------------------------------------------------------------------------------

import pandas_ta as ta
from .strategy import strategy
from . import register_strategy

@register_strategy("RSI_Bollinger")
class RSI_Bollinger(strategy):
    
    def __init__(self, 
                 name="RSI_Bollinger",
                 rsi_length: int = 14,      # RSI 週期~
                 bb_length: int = 20,       # 布林通道週期
                 bb_std: float = 2.0,       # 布林通道標準差倍數
                 rsi_oversold: int = 30,    # 超賣門檻 (做多)
                 rsi_overbought: int = 70): # 超買門檻 (做空)
        
        super().__init__(name)
        self.rsi_length = rsi_length
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, df):
        # 1. 計算 RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_length)
        
        # 2. 計算布林通道 (Bollinger Bands)
        # pandas_ta 的 bbands 回傳順序固定為: [Lower, Mid, Upper, Bandwidth, Percent]
        bb = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
        
        # [修正] 使用 iloc 依據位置取值，避免因為 "2.0" vs "2" 的命名問題導致報錯
        # 確保 bb 有計算出來
        if bb is not None and not bb.empty:
            df['bb_lower'] = bb.iloc[:, 0] # 取第 0 欄 (Lower Band)
            df['bb_upper'] = bb.iloc[:, 2] # 取第 2 欄 (Upper Band)
        else:
            # 防呆：如果資料不足導致無法計算
            df['bb_lower'] = df['close']
            df['bb_upper'] = df['close']

        # 3. 初始化信號
        df["signal"] = 0
        
        # 4. 進場邏輯
        # 做多：收盤價 < 布林下軌 且 RSI < 超賣區
        long_condition = (df['close'] < df['bb_lower']) & (df['rsi'] < self.rsi_oversold)
        
        # 做空：收盤價 > 布林上軌 且 RSI > 超買區
        short_condition = (df['close'] > df['bb_upper']) & (df['rsi'] > self.rsi_overbought)
        
        # 5. 寫入信號
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1
        
        # 記錄策略名稱
        df["strategy_name"] = self.name
        
        return df