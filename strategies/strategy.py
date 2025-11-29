# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案定義了策略的「介面 (Interface)」。
# 它的功能是強制規範所有具體的策略 (如 MACD策略, RSI策略) 都必須遵守同一套標準。
# 這樣做的好處是，主程式 (main.py) 不需要知道你是什麼策略，只要知道你是「strategy 的小孩」，
# 就能確定你一定有 generate_signal 方法可以呼叫。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 從 Python 標準函式庫 abc (Abstract Base Classes) 引入 ABC 和 abstractmethod。
# ABC:這是一個輔助類別，讓我們的 class 繼承它之後變成「抽象類別」。
# abstractmethod:這是一個裝飾器 (Decorator)，用來標記某個方法是「抽象方法」，
# 意思是：「我這裡不寫實作，但我規定我的子類別一定要寫，不然就會報錯」。
from abc import ABC, abstractmethod

# [Class 說明]
# 職責：所有策略的父類別 (Parent Class)。
# 這是一個抽象類別，你不能直接實例化它 (不能寫 s = strategy())，只能繼承它。
class strategy(ABC):
    """所有策略必須繼承這個父類別，保證方法一致"""

    # [Function 說明]
    # 功能：建構子，初始化策略物件。
    # 原理：接收一個名稱參數，並存到物件屬性中。
    def __init__(self, name):
        # 將傳入的策略名稱 (name) 儲存為實例變數 (self.name)，方便辨識。
        self.name = name
    
    # [Function 說明]
    # 功能：定義產生交易訊號的方法介面。
    # 原理：加上 @abstractmethod 後，這個方法變成了一個「契約」。
    # 任何繼承 strategy 的子類別，都 *必須* 實作這個方法，否則程式無法執行。
    # 參數 df: 預期會接收一個包含 K 線資料的 Pandas DataFrame。
    @abstractmethod
    def generate_signal(self, df):
        """signal = 0: none, 1 : buy, -1 : sell"""
        # 如果子類別忘了實作這個方法，卻嘗試呼叫它，就會拋出這個錯誤，提醒開發者去補寫。
        # (實際上，因為繼承了 ABC，如果沒實作這個方法，連物件都建立不起來，這是雙重保險)
        raise NotImplementedError("此策略還沒有signal(data)")
    
    # [Function 說明]
    # 功能：定義物件被轉成字串時的顯示方式。
    # 原理：當我們使用 print(strategy_object) 時，Python 會自動呼叫這個 __str__ 方法。
    def __str__(self):
        # 直接回傳策略的名稱，這樣印出來比較好讀 (例如印出 "MACD_Strategy" 而不是 "<object at 0x...>")。
        return self.name