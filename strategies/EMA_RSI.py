# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案定義了一個名為 "EMA_RSI" 的具體交易策略。
# 它是 `strategy` 父類別的子類別 (Child Class)。
# 它的核心功能是實作 `generate_signal` 方法，根據 EMA 和 RSI 指標計算買賣訊號。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 引入 pandas_ta 套件，並簡寫為 ta。
# 這是一個專門為 Pandas 設計的技術分析庫，能快速計算數百種指標 (如 RSI, MACD, 布林通道)。
import pandas_ta as ta

# 從目前的套件 (即 strategies 資料夾) 中的 strategy.py 引入 strategy 父類別。
# 這是為了讓我們的類別能繼承它，遵守統一的介面規範。
from .strategy import strategy

# 從目前的套件 (即 strategies 資料夾) 的 __init__.py 引入 register_strategy 裝飾器。
# 這個裝飾器的用途是將寫好的策略「註冊」到系統中，讓 main.py 可以透過字串名稱 ("EMA_RSI") 找到這個類別。
from . import register_strategy

# [Class 說明]
# 職責：實作 EMA + RSI 的交易邏輯。
# 裝飾器 @register_strategy("EMA_RSI") 會在程式啟動時自動把這個 class 存到 STRATEGY_REGISTRY 字典裡。
@register_strategy("EMA_RSI")
class ema_RSI(strategy):
    
    # [Function 說明]
    # 功能：初始化策略參數。
    # 原理：設定計算指標所需的長度 (如 EMA 30, RSI 9) 以及觸發訊號的門檻值 (Lower/Upper)。
    # 這些參數都有預設值，也可以從外部傳入修改。
    def __init__(self,
                 name="EMA_RSI",          # 策略名稱
                 ema_length:int|None = 30,# EMA 的週期長度，預設看過去 30 根 K 棒
                 rsi_length:int|None = 9, # RSI 的週期長度，預設 9
                 lower:int|None = 30,     # RSI 低檔區 (超賣區) 門檻
                 upper:int|None = 70):    # RSI 高檔區 (超買區) 門檻
        # 呼叫父類別 (strategy) 的建構子，確保基礎屬性 (如 self.name) 被正確設定。
        super().__init__(name)
        # 將傳入的參數儲存到物件屬性中，供後續 generate_signal 使用。
        self.ema_length = ema_length
        self.rsi_length = rsi_length
        self.lower = lower
        self.upper = upper

    # [Function 說明]
    # 功能：核心交易邏輯。
    # 原理：
    # 1. 使用 pandas_ta 計算 EMA 和 RSI 數值。
    # 2. 判斷買入條件：價格 > EMA (趨勢向上) 且 RSI > 70 (動能強勁)。
    # 3. 判斷賣出條件：價格 < EMA (趨勢向下) 且 RSI < 30 (動能疲弱)。
    # 4. 使用 Pandas 的向量化操作 (.loc) 快速標記訊號，避免用慢速的 for 迴圈。
    def generate_signal(self, df):
        # 使用 pandas_ta 計算 EMA 指標。
        # df['close'] 是收盤價序列，self.ema_length 是週期 (30)。
        ema = ta.ema(df['close'],self.ema_length)
        
        # 使用 pandas_ta 計算 RSI 指標。
        rsi = ta.rsi(df['close'], self.rsi_length)
        
        # 將計算出來的 EMA 數據存回 DataFrame 的新欄位 "ema"。
        # 這樣我們可以在最後輸出的 CSV 中看到指標數值，方便驗證。
        df["ema"] = ema
        
        # 將計算出來的 RSI 數據存回 DataFrame 的新欄位 "rsi"。
        df["rsi"] = rsi
        
        # 初始化 "signal" 欄位，預設全部填 0 (無訊號/觀望)。
        df["signal"] = 0
        
        # [買入邏輯 - 順勢動能策略]
        # 條件 1: 收盤價 > EMA (代表目前處於上升趨勢)。
        # 條件 2: RSI > Upper (通常是 70)。
        # 注意：一般教科書說 RSI > 70 是超買要賣，但這裡採用的是「動能 (Momentum)」邏輯，
        # 認為 RSI 衝破 70 代表買盤極強，趨勢會延續，所以做多。
        # 使用 .loc[條件, 欄位] = 值 的語法進行賦值。
        df.loc[ (df["close"] > df["ema"]) & (df["rsi"] > self.upper), "signal" ] = 1
        
        # [賣出邏輯 - 順勢動能策略]
        # 條件 1: 收盤價 < EMA (代表目前處於下降趨勢)。
        # 條件 2: RSI < Lower (通常是 30)。
        # 同理，這裡認為 RSI 跌破 30 代表殺盤極強，會繼續跌，所以做空。
        df.loc[ (df["close"] < df["ema"]) & (df["rsi"] < self.lower), "signal" ] = -1
        
        # 在 DataFrame 中記錄這份資料是用哪個策略算出來的，方便後續追蹤。
        df["strategy_name"] = self.name
        
        # 回傳處理好的 DataFrame (包含 signal 欄位)。
        return df