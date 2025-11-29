import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_config():
    """提供一個假的 config 物件，用於測試"""
    return {
        "基本設定": {
            "strategy": "TEST_STRATEGY",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "testnet": False,
            "testnet_url": "http://fake.url",
            "use_mark_price_kline": False,
            "fetch_limit": 100,
            "sleep_time": 0.1
        },
        "回測設定": {
            "start_time": "2024-01-01 00:00",
            "end_time": "2024-01-02 00:00"
        }
    }

@patch('main.DerivativesTradingUsdsFutures')
@patch('main.to_timestamp', return_value=123456789)
@patch('main.data_to_csv')
@patch('main.get_kline_data')
@patch('main.load_strategy')
@patch('main.load_config')
def test_main_flow(mock_load_config, mock_load_strategy, mock_get_kline_data, mock_data_to_csv, mock_to_timestamp, mock_api_client, mock_config):
    """測試 main.py 的主要執行流程"""
    # 由於 main.py 已經重構，我們可以安全地導入它
    import main as main_module

    # --- 準備 (Arrange) ---
    # 設定 Mock 物件的回傳值
    mock_load_config.return_value = mock_config
    
    # 模擬策略物件和它的方法
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.generate_signal.return_value = pd.DataFrame({'signal': [1]})
    mock_load_strategy.return_value = mock_strategy_instance

    # 模擬 get_kline_data 回傳一個 DataFrame 和 needsave=False
    mock_get_kline_data.return_value = (pd.DataFrame({'close': [100]}), False)

    # --- 執行 (Act) ---
    main_module.main()

    # --- 驗證 (Assert) ---
    mock_load_config.assert_called_once()
    mock_load_strategy.assert_called_with("TEST_STRATEGY")
    mock_get_kline_data.assert_called_once()
    mock_strategy_instance.generate_signal.assert_called_once()
    # 驗證最後儲存訊號檔的呼叫
    mock_data_to_csv.assert_called_with(mock_strategy_instance.generate_signal.return_value, is_raw=False)