# =========================================================================================
# 檔案名稱：strategies/EMA_RSI.py
# 檔案說明：
#   定義了一個名為 "EMA_RSI" 的具體交易策略。
#   該策略結合了趨勢指標 (EMA) 與動能指標 (RSI)，屬於順勢突破型的策略。
#
# 繼承關係：
#   EMA_RSI (Child) -> strategy (Parent, defined in strategy.py)
#   必須實作父類別規定的 `generate_signal` 方法。
# =========================================================================================

# [模組引用解析]
# 引入 pandas_ta 套件，這是專為 DataFrame 設計的技術分析庫，能進行高效的向量化計算。
import pandas_ta as ta

# 從目前的套件 (即 strategies 資料夾) 中的 strategy.py 引入 strategy 父類別。
# 這是為了讓我們的類別能繼承它，遵守統一的介面規範。
from .strategy import strategy

# 從目前的套件 (即 strategies 資料夾) 的 __init__.py 引入 register_strategy 裝飾器。
# 用途：將此策略類別「註冊」到全域字典中，讓主程式能透過字串 "EMA_RSI" 動態載入它。
from . import register_strategy

# =========================================================================================
# 類別：ema_RSI
# 職責：實作 EMA + RSI 的交易邏輯
# 說明：
#   使用 @register_strategy 裝飾器，確保程式啟動時自動註冊。
# =========================================================================================
@register_strategy("EMA_RSI")
class ema_RSI(strategy):
    
    # [方法：初始化]
    # 輸入：
    #   - name: 策略名稱。
    #   - ema_length: EMA 移動平均線的週期 (Trend Filter)。
    #   - rsi_length: RSI 相對強弱指標的週期 (Momentum)。
    #   - lower/upper: RSI 的超賣/超買界線。
    # 邏輯：
    #   將策略參數儲存為物件屬性，供後續 generate_signal 使用。
    #   這些參數的預設值可以被 Optimizer (優化器) 覆蓋。
    def __init__(self,
                 name="EMA_RSI",          # 策略名稱
                 ema_length:int|None = 30,# EMA 的週期長度，預設看過去 30 根 K 棒
                 rsi_length:int|None = 9, # RSI 的週期長度，預設 9
                 lower:int|None = 30,     # RSI 低檔區 (超賣區) 門檻
                 upper:int|None = 70):    # RSI 高檔區 (超買區) 門檻
        # 呼叫父類別 (strategy) 的建構子，確保基礎屬性 (如 self.name) 被正確設定。
        super().__init__(name)
        # 將傳入的參數儲存到物件屬性中。
        self.ema_length = ema_length
        self.rsi_length = rsi_length
        self.lower = lower
        self.upper = upper

    # [方法：產生訊號]
    # 輸入：
    #   - df: 原始 K 線資料 DataFrame (包含 open, high, low, close)。
    # 輸出：
    #   - df: 增加 "signal" 欄位後的 DataFrame。
    # 邏輯：
    #   1. 計算指標：利用 pandas_ta 計算 EMA 與 RSI。
    #   2. 向量化判斷：不使用迴圈，而是利用 Pandas 的條件篩選語法 (.loc) 一次性標記所有訊號。
    def generate_signal(self, df):
        # 使用 pandas_ta 計算 EMA 指標。
        # df['close'] 是收盤價序列，self.ema_length 是週期 (30)。
        ema = ta.ema(df['close'],self.ema_length)
        
        # 使用 pandas_ta 計算 RSI 指標。
        rsi = ta.rsi(df['close'], self.rsi_length)
        
        # [變數追蹤] 指標儲存
        # 將計算出來的數據存回 DataFrame，這對於除錯與圖表繪製非常重要。
        df["ema"] = ema
        df["rsi"] = rsi
        
        # 初始化 "signal" 欄位，預設全部填 0 (無訊號/觀望)。
        df["signal"] = 0
        
        # [控制流解析] 買入邏輯 (順勢動能)
        # 條件 1: 收盤價 > EMA (代表目前處於上升趨勢)。
        # 條件 2: RSI > Upper (代表買氣強勁，動能突破)。
        # 注意：這裡採用的是「強者恆強」的動能策略，而非傳統的 RSI 超買反轉。
        # 使用 .loc[條件, 欄位] = 值 的語法進行高效賦值。 # 註1
        df.loc[ (df["close"] > df["ema"]) & (df["rsi"] > self.upper), "signal" ] = 1
        
        # [控制流解析] 賣出邏輯 (順勢動能)
        # 條件 1: 收盤價 < EMA (代表目前處於下降趨勢)。
        # 條件 2: RSI < Lower (代表賣壓沉重，動能崩潰)。
        df.loc[ (df["close"] < df["ema"]) & (df["rsi"] < self.lower), "signal" ] = -1
        
        # 在 DataFrame 中記錄這份資料是用哪個策略算出來的，方便後續追蹤。
        df["strategy_name"] = self.name
        
        # 回傳處理好的 DataFrame (包含 signal 欄位)。
        return df

# ====== 備註區 ======
# 註1: 條件語法的效能
#      使用 df.loc[(cond1) & (cond2)] 是 Pandas 中效能最好的寫法之一。
#      請避免使用 df.apply(lambda x: ...) 逐行計算，那樣會慢上數十倍。