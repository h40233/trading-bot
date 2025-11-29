# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案是策略套件的初始化檔案，負責建立一個「全域策略註冊表」。
# 它的核心功能有兩個：
# 1. 定義 `STRATEGY_REGISTRY` 字典，用來存放所有可用的策略 (名稱 -> 類別)。
# 2. 定義 `register_strategy` 裝飾器，讓個別策略檔案可以把自己註冊進來。
# 3. 自動掃描並 import 這個資料夾下的所有 .py 檔案，觸發註冊動作。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 引入 Type 和 Dict，這是 Python 的型別提示 (Type Hinting) 工具。
# 它們不會影響程式執行，但能讓程式碼更易讀，並讓 IDE (如 VS Code) 提供更好的自動補全。
from typing import Type, Dict

# 引入我們定義的 strategy 父類別，用於型別檢查，確保註冊進來的都是合法的策略類別。
# 注意：這裡的 .strategy 代表從當前目錄下的 strategy.py 引入。
from .strategy import strategy

# 引入 pathlib，這是處理檔案路徑的現代化工具，比舊的 os.path 好用。
from pathlib import Path

# 引入 pkgutil 和 importlib，這是 Python 的動態載入模組工具。
# 我們需要它們來「自動掃描」資料夾裡有哪些檔案，並把它們 import 進來。
import pkgutil, importlib


# [變數說明]
# 這是全域註冊表 (Global Registry)。
# 它的結構是一個字典：Key 是策略名稱 (字串)，Value 是策略類別 (Class)。
# 例子：{ "EMA_RSI": <class 'strategies.EMA_RSI.ema_RSI'> }
STRATEGY_REGISTRY: Dict[str, Type[strategy]] = {}

# [Function 說明]
# 功能：這是一個「裝飾器工廠 (Decorator Factory)」。
# 原理：它接收一個參數 (策略名稱)，然後回傳一個真正的裝飾器函式。
# 用途：放在策略 class 的上方，例如 @register_strategy("MyStrategy")，
# 這樣當 Python 讀到那個 class 時，就會自動執行這個函式，把 class 存入 STRATEGY_REGISTRY。
def register_strategy(name:str):
    """
    裝飾器：把策略類別註冊到全域字典裡
    用法：@register_strategy("ma_cross")
    """
    # 這是真正的裝飾器函式，它接收被裝飾的 Class (cls)。
    def decorator(cls: Type[strategy]):
        # 檢查名稱是否重複，避免兩個策略用了同一個名字互相覆蓋。
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"策略名稱 '{name}' 已經被註冊過了")
        
        # 核心動作：將 名稱 與 類別 的對應關係存入全域字典。
        STRATEGY_REGISTRY[name] = cls
        
        # 裝飾器必須把原本的 class 回傳回去，這樣程式其他地方才能正常使用這個 class。
        return cls
    # 回傳內部的裝飾器函式。
    return decorator


# 👇 自動掃描 strategies 資料夾並 import
# [程式邏輯說明]
# 這段程式碼在 `import strategies` 時會自動執行。它的目的是把同資料夾下的所有 .py 檔跑一遍。

# 1. 取得當前檔案 (__init__.py) 所在的資料夾路徑。
# Path(__file__) 是這個檔案本身，.parent 就是它所在的資料夾 (即 strategies 資料夾)。
package_dir = Path(__file__).resolve().parent

# 2. 使用 pkgutil.iter_modules 掃描該資料夾下的所有模組。
# 它會回傳一個迭代器，我們只需要中間的 module_name (檔名，不含 .py)。
for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        
    # 3. 過濾掉基礎類別和初始化檔案。
    # "strategy" 是父類別定義檔，"__init__" 是自己，這兩個不需要被當作策略載入。
    if module_name not in ("strategy", "__init__"):  # 避免重複載入基底類
        
        # 4. 動態 import 模組。
        # f"{__name__}.{module_name}" 會變成例如 "strategies.EMA_RSI"。
        # 當這行執行時，Python 會去讀取 EMA_RSI.py，
        # 這會導致 EMA_RSI.py 裡面的 `@register_strategy` 被執行，
        # 進而將該策略自動加入 `STRATEGY_REGISTRY` 中。
        importlib.import_module(f"{__name__}.{module_name}")