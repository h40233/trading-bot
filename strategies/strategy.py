# =========================================================================================
# 檔案名稱：strategies/strategy.py
# 檔案說明：
#   定義了策略的「抽象基類 (Abstract Base Class)」。
#   這份檔案是策略模組的「憲法」，它強制規定所有子策略必須具備的行為（如產生訊號）。
#
# 設計模式：
#   採用「樣板方法模式 (Template Method Pattern)」的精神，
#   父類別定義介面，子類別負責具體實作。
# =========================================================================================

# [模組引用解析]
# 從 Python 標準函式庫 abc 引入 ABC (Abstract Base Class) 與 abstractmethod。
# 用途：建立無法被直接實例化的抽象類別，並強制子類別實作特定方法。
from abc import ABC, abstractmethod

# =========================================================================================
# 類別：strategy
# 職責：所有策略的父類別
# 說明：
#   繼承 ABC 代表這是一個抽象類別。
#   你不能寫 `s = strategy()`，必須寫 `class MyStrat(strategy): ...`。
# =========================================================================================
class strategy(ABC):
    """所有策略必須繼承這個父類別，保證方法一致"""

    # [方法：初始化]
    # 邏輯：接收並儲存策略名稱。
    def __init__(self, name):
        # 將傳入的策略名稱 (name) 儲存為實例變數，方便日誌記錄與識別。
        self.name = name
    
    # [抽象方法：產生訊號]
    # 裝飾器 @abstractmethod 代表這是一個「契約」。
    # 任何繼承此類別的子類別，都 *必須* 實作這個方法，否則 Python 會拒絕執行。
    # 參數 df: 預期接收一個包含 K 線資料 (Open, High, Low, Close) 的 DataFrame。
    @abstractmethod
    def generate_signal(self, df):
        """signal = 0: none, 1 : buy, -1 : sell"""
        # 若子類別忘記實作此方法，這裡會拋出錯誤提醒開發者。
        raise NotImplementedError("此策略還沒有signal(data)")
    
    # [方法：字串表示]
    # 邏輯：定義物件被 print() 時的顯示格式。
    def __str__(self):
        # 回傳策略名稱，讓除錯訊息更友善 (例如顯示 "MACD_Strategy" 而非 "<object at 0x...>")。
        return self.name

# ====== 備註區 ======
# 註1: 為什麼要用 ABC？
#      在大型專案中，確保所有策略都有 `generate_signal` 方法至關重要。
#      若不使用 ABC，開發者可能會拼錯方法名稱 (如 `gen_signal`)，導致回測引擎呼叫失敗。
#      ABC 能在程式啟動階段就攔截這類錯誤。