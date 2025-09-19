from typing import Type, Dict
from .strategy import strategy
from pathlib import Path
import pkgutil, importlib


# 全域註冊表
STRATEGY_REGISTRY: Dict[str, Type[strategy]] = {}

def register_strategy(name:str):
    """
    裝飾器：把策略類別註冊到全域字典裡
    用法：@register_strategy("ma_cross")
    """
    def decorator(cls: Type[strategy]):
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"策略名稱 '{name}' 已經被註冊過了")
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


# 👇 自動掃描 strategies 資料夾並 import
package_dir = Path(__file__).resolve().parent
for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
    if module_name not in ("strategy", "__init__"):  # 避免重複載入基底類
        importlib.import_module(f"{__name__}.{module_name}")