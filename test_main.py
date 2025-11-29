# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是一個單元測試檔案，專門用來測試 `main.py` 的主程式流程。
# 它的核心架構遵循 AAA 原則：Arrange (準備)、Act (執行)、Assert (驗證)。
# 為了避免測試時真的連線到交易所或讀寫檔案，這裡大量使用了 `unittest.mock` 來「模擬」外部依賴。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 引入 pytest，這是 Python 功能最強大且最流行的測試框架。
# 我們用它來定義測試案例 (test cases) 和測試夾具 (fixtures)。
import pytest

# 引入 pandas。
# 在測試中，我們需要憑空捏造一些假的 DataFrame 資料，用來模擬從 API 抓回來的 K 線。
import pandas as pd

# 從 unittest 標準庫引入 patch 和 MagicMock。
# patch: 用來把原本程式裡的某個功能「暫時替換」成假的。
# MagicMock: 一個萬能的假物件，可以模擬任何 class 或 function 的行為 (例如假裝自己被呼叫了幾次)。
from unittest.mock import patch, MagicMock

# [Function 說明]
# 功能：這是一個 Pytest Fixture (測試夾具)。
# 原理：Fixture 的作用是提供測試所需的「前置準備資料」。在這裡，它回傳一個字典，模擬真實的 config.json 內容。
# 用途：這樣我們就不需要在每個測試函式裡重複寫這個字典，且確保測試用的設定是固定的。
@pytest.fixture
def mock_config():
    """提供一個假的 config 物件，用於測試"""
    # 回傳一個字典 (Dictionary)，結構完全模仿 config.json。
    return {
        "基本設定": {
            "strategy": "TEST_STRATEGY", # 設定測試用的策略名稱
            "symbol": "BTCUSDT",         # 設定測試用的幣種
            "timeframe": "1h",           # 設定時間週期
            "testnet": False,            # 設定不使用測試網
            "testnet_url": "http://fake.url", # 假的 URL
            "use_mark_price_kline": False,    # 不使用標記價格
            "fetch_limit": 100,               # 限制抓取筆數
            "sleep_time": 0.1                 # 模擬的暫停時間
        },
        "回測設定": {
            "start_time": "2024-01-01 00:00", # 模擬的回測開始時間
            "end_time": "2024-01-02 00:00"    # 模擬的回測結束時間
        }
    }

# [Function 說明]
# 功能：測試 main.py 的主邏輯是否正確串接各個步驟。
# 原理：使用 `@patch` 裝飾器將 main.py 裡用到的外部依賴 (API、檔案讀寫等) 替換成 Mock 物件。
# 這樣我們就可以在沒有網路、沒有檔案的情況下，驗證程式邏輯 (Logic) 是否正確。
# 參數解釋：每個 @patch 對應一個參數，順序是由下往上對應 (由內而外)。
@patch('main.DerivativesTradingUsdsFutures') # 1. 模擬 Binance API 客戶端，對應 mock_api_client
@patch('main.to_timestamp', return_value=123456789) # 2. 模擬時間轉換函式，直接讓它回傳固定數字，對應 mock_to_timestamp
@patch('main.data_to_csv') # 3. 模擬存檔函式，避免真的寫入硬碟，對應 mock_data_to_csv
@patch('main.get_kline_data') # 4. 模擬抓 K 線函式，對應 mock_get_kline_data
@patch('main.load_strategy') # 5. 模擬載入策略函式，對應 mock_load_strategy
@patch('main.load_config') # 6. 模擬讀取設定檔函式，對應 mock_load_config
def test_main_flow(mock_load_config, mock_load_strategy, mock_get_kline_data, mock_data_to_csv, mock_to_timestamp, mock_api_client, mock_config):
    """測試 main.py 的主要執行流程"""
    # 在測試函式內部匯入 main 模組。
    # 這是為了確保我們是在 patch 設定好之後才載入模組，或者是為了避免循環匯入的問題。
    import main as main_module

    # --- 準備 (Arrange) ---
    # 設定 Mock 物件的行為與回傳值。
    
    # 設定 mock_load_config 被呼叫時，要回傳我們上面定義的 mock_config fixture 資料。
    # 這樣 main.py 執行時就會以為讀到了真的設定檔。
    mock_load_config.return_value = mock_config
    
    # 模擬策略物件和它的方法。
    # 建立一個 MagicMock 物件來代表「策略實例 (Instance)」。
    mock_strategy_instance = MagicMock()
    # 設定這個假策略的 `generate_signal` 方法被呼叫時，回傳一個假的 DataFrame。
    # 這模擬了策略計算出訊號的結果。
    mock_strategy_instance.generate_signal.return_value = pd.DataFrame({'signal': [1]})
    # 設定 load_strategy 被呼叫時，回傳上面這個假策略物件。
    mock_load_strategy.return_value = mock_strategy_instance

    # 模擬 get_kline_data 的回傳值。
    # 它原本會回傳 (DataFrame, bool)，我們這裡模擬回傳一個簡單的 DataFrame 和 False。
    # close: [100] 是隨便給的假數據，False 代表不需要存 raw data (模擬邏輯)。
    mock_get_kline_data.return_value = (pd.DataFrame({'close': [100]}), False)

    # --- 執行 (Act) ---
    # 實際執行 main.py 的 main() 函式。
    # 因為所有的外部依賴都被 Mock 取代了，這裡跑起來會非常快，且不會有副作用。
    main_module.main()

    # --- 驗證 (Assert) ---
    # 驗證 load_config 是否被呼叫了一次。如果沒被呼叫，代表程式流程有錯。
    mock_load_config.assert_called_once()
    
    # 驗證 load_strategy 是否被呼叫，且參數必須是 "TEST_STRATEGY" (來自 mock_config)。
    # 這確保程式有正確根據設定檔載入對應的策略。
    mock_load_strategy.assert_called_with("TEST_STRATEGY")
    
    # 驗證 get_kline_data 是否被呼叫了一次。確保有去抓資料的動作。
    mock_get_kline_data.assert_called_once()
    
    # 驗證策略物件的 generate_signal 方法是否被呼叫了一次。確保有執行策略運算。
    mock_strategy_instance.generate_signal.assert_called_once()
    
    # 驗證最後儲存訊號檔的呼叫。
    # 檢查 data_to_csv 是否被呼叫，且傳入的第一個參數是策略產生的假 DataFrame，is_raw 參數是 False。
    # 這確保程式最後有嘗試把結果存檔。
    mock_data_to_csv.assert_called_with(mock_strategy_instance.generate_signal.return_value, is_raw=False)