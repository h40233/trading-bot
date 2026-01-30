# =========================================================================================
# 檔案名稱：strategies/__init__.py
# 檔案說明：
#   策略套件的初始化檔案，負責建立「全域策略註冊表」。
#   不同於 os.walk 或 glob 的寫法，此版本使用了更為嚴謹的 pkgutil 與 pathlib
#   來實現自動化的模組掃描與註冊。
#
# 核心功能：
#   1. 定義 Type Hinting (型別提示)，提升程式碼可讀性與 IDE 支援度。
#   2. 實作 `register_strategy` 裝飾器，供個別策略檔案呼叫。
#   3. 初始化時自動掃描當前目錄，將所有策略模組動態匯入。
# =========================================================================================

# [Import 說明]
# 引入 Type 與 Dict 用於型別標註，這不會影響執行，但在大型專案中對靜態分析非常有幫助。
from typing import Type, Dict

# [Import 說明]
# 從同一層級的 strategy.py 引入基礎策略類別，用於限制註冊的型別必須是 strategy 的子類。
from .strategy import strategy

# [Import 說明]
# 引入 pathlib 的 Path 物件，這是現代 Python 處理檔案路徑的標準做法，比 os.path 更直觀且物件導向。
from pathlib import Path

# [Import 說明]
# pkgutil: 用於管理與遍歷 Python 套件的工具，能更準確地找出可匯入的模組。
# importlib: 用於動態載入模組 (Dynamic Import)。
import pkgutil, importlib


# [變數追蹤] STRATEGY_REGISTRY
# 意義：全域策略註冊表，儲存 "策略名稱" 對應 "策略類別" 的映射。
# 型別：Key 為字串，Value 為繼承自 strategy 的類別 (Type[strategy])。
STRATEGY_REGISTRY: Dict[str, Type[strategy]] = {}

# [函式說明]
# 功能：策略註冊裝飾器 (Decorator Factory)。
# 輸入：name (策略在系統中的唯一識別碼)。
# 輸出：回傳一個裝飾器函式 (decorator)。
def register_strategy(name:str):
    """
    裝飾器：把策略類別註冊到全域字典裡
    用法：@register_strategy("ma_cross")
    """
    # [內部函式] 實際的裝飾器
    # 參數 cls: 被裝飾的策略類別。
    def decorator(cls: Type[strategy]):
        # [邏輯] 防呆檢查
        # 確保不會有兩個策略使用相同的名稱，避免後蓋前導致邏輯錯誤。
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"策略名稱 '{name}' 已經被註冊過了")
        
        # [核心邏輯] 註冊
        # 將名稱與類別存入全域字典。
        STRATEGY_REGISTRY[name] = cls
        
        # 回傳原始類別，確保該類別仍可被正常實例化。
        return cls
    # 回傳閉包 (Closure)。
    return decorator


# [變數追蹤] package_dir
# 意義：當前檔案 (__init__.py) 所在的資料夾路徑。
# 邏輯：使用 Path(__file__) 取得當前檔案路徑，resolve() 轉為絕對路徑，parent 取得父目錄。
package_dir = Path(__file__).resolve().parent

# [控制流解析] 自動掃描與匯入
# 使用 pkgutil.iter_modules 遍歷指定目錄下的所有模組。
# 它會回傳 (module_finder, name, ispkg) 的 Tuple，我們只關心 name (模組名稱)。
for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        
    # [邏輯] 排除非策略檔案
    # "strategy" 是介面定義檔，"__init__" 是本檔案，不應被視為交易策略載入。
    if module_name not in ("strategy", "__init__"):  # 避免重複載入基底類
        
        # [核心邏輯] 動態匯入
        # 使用 importlib.import_module 執行匯入動作。
        # f"{__name__}.{module_name}" 會解析為例如 "strategies.EMA_RSI"。
        # 當模組被匯入時，其內部的 @register_strategy 裝飾器會自動執行，完成註冊。
        importlib.import_module(f"{__name__}.{module_name}")

# ====== 備註區 ======
# 註1: pkgutil vs glob
#      相比於使用 glob.glob("*.py")，pkgutil.iter_modules 是更標準的做法。
#      它能正確處理 Python 的模組查找規則 (包含 .pyc, .zip 等情況)，且不會誤讀非模組的檔案。