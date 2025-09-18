from typing import Type, Dict
from .strategy import strategy


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