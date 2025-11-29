# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是程式的「主入口 (Main Entry Point)」，負責協調各個模組運作。
# 它的核心流程如下：
# 1. 初始化環境 (Logging, Config)。
# 2. 建立與 Binance 交易所的連線 (Client)。
# 3. 抓取 K 線歷史資料 (K-line Data)。
# 4. 呼叫策略計算交易訊號 (Signal Generation)。
# 5. 將原始資料與計算結果儲存為 CSV 檔案。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 從 Binance 官方 SDK 引入必要的類別。
# DerivativesTradingUsdsFutures: 用於建立 U本位合約交易的主要客戶端物件。
# ConfigurationRestAPI: 用於設定 API 金鑰、URL 等連線參數的物件。
# DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL: 官方正式環境 (Mainnet) 的網址常數。
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (DerivativesTradingUsdsFutures,ConfigurationRestAPI,DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL)

# 引入 logging 模組，用於顯示程式執行的進度與狀態 (如 INFO 訊息)。
import logging

# 從 util.py 引入所有工具函式 (如 load_config, get_kline_data 等)。
# 注意：使用 `import *` 雖然方便，但在大型專案中可能會造成命名衝突，需謹慎使用。
# 這裡也隱含引入了 util 裡面的 os, pandas 等模組，所以下面可以直接用 os.getenv。
from util import *

# [Function 說明]
# 功能：主執行函式，程式的起點。
# 原理：依序執行 設定讀取 -> 建立連線 -> 抓取資料 -> 策略運算 -> 儲存結果 的流水線作業。
def main():
    """主執行函式"""
    #logging設定
    # 設定日誌輸出的等級為 INFO。這表示程式會印出一般資訊 (如「載入成功」、「抓取完成」) 及其以上等級的訊息。
    logging.basicConfig(level=logging.INFO)

    #抓config
    # 呼叫 util.py 中的 load_config 函式，讀取 config.json 檔案，並轉為 Python 字典。
    # 這讓我們可以透過修改 JSON 檔來改變程式行為，而不用改程式碼。
    config = load_config()

    #載入策略
    # 根據 config.json 中 "基本設定" -> "strategy" 指定的名字 (如 "MACD_Strategy")，
    # 去 util.py (透過 strategies 模組) 動態載入對應的策略類別並實例化。
    strategy = load_strategy(config["基本設定"]["strategy"])

    #設定datetime
    # 將 config 中人類可讀的時間字串 (如 "2024-01-01 00:00") 轉換成 API 需要的 Unix Timestamp (毫秒)。
    start_time = to_timestamp(config["回測設定"]["start_time"])
    end_time = to_timestamp(config["回測設定"]["end_time"])

    #建立client
    # 從環境變數 (.env 檔) 中讀取 API KEY 和 SECRET。
    # 這裡使用 os.getenv 是資安最佳實踐，避免將密碼直接寫死在程式碼中 (Hardcoding)。
    # 第二個參數 "" 是預設值，如果找不到環境變數就回傳空字串。
    API_KEY = os.getenv("API_KEY","")
    API_SECRET = os.getenv("API_SECRET","")
    
    # 決定要連線到「測試網 (Testnet)」還是「正式網 (Mainnet)」。
    # 使用 Python 的三元運算子 (Ternary Operator)：如果 config 設定 testnet 為 True，就用 testnet_url，否則用正式網址。
    url = config["基本設定"]["testnet_url"] if config["基本設定"]["testnet"] else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL

    # 建立 API 設定物件。
    # 將金鑰、密碼和網址封裝進 ConfigurationRestAPI 物件中。
    config_api=ConfigurationRestAPI(
                api_key = API_KEY,      # 設定 API 公鑰
                api_secret = API_SECRET,# 設定 API 私鑰 (用於簽名)
                base_path = url         # 設定連線目標網址
                )
    # 使用上面的設定，建立主要的交易客戶端 (client)。
    # .rest_api 屬性是 SDK 中用來發送 RESTful 請求的實際介面。
    client = DerivativesTradingUsdsFutures(config_rest_api=config_api).rest_api

    #抓K線
    # 呼叫 util.py 的 get_kline_data 函式向交易所抓資料。
    # 參數包含了：客戶端物件、幣種、時間週期、開始/結束時間、是否用標記價格、單次抓取限制、以及防封鎖的暫停時間。
    # 回傳值：
    # df: 包含 K 線數據的 Pandas DataFrame。
    # needsave: 一個布林值，指示這份資料是否是新抓下來的 (如果是新的，後面才需要存檔)。
    df, needsave = get_kline_data(client, config["基本設定"]["symbol"], config["基本設定"]["timeframe"], start_time, end_time, config["基本設定"]["use_mark_price_kline"],config["基本設定"]["fetch_limit"],config["基本設定"]["sleep_time"])
    
    # 如果 needsave 為 True (代表資料是剛從 API 抓下來的，硬碟裡原本沒有)。
    if needsave:
        # 呼叫 util.py 的 data_to_csv 將「原始資料」存檔。
        # is_raw=True 會讓檔名標記為原始資料，並存到 data/raw 資料夾。
        data_to_csv(df,is_raw = True)
    
    # 呼叫策略物件的 generate_signal 方法。
    # 這一步是核心：將 K 線資料 (df) 傳入策略，策略會計算技術指標 (如 MA, RSI) 並產出 "signal" 欄位。
    # out 是一個包含了原始資料 + 技術指標 + 交易訊號的新 DataFrame。
    out = strategy.generate_signal(df)
    
    # 將「處理後 (含訊號)」的資料存檔。
    # is_raw=False 會讓檔名標記策略名稱，並存到 data/processed 資料夾。
    data_to_csv(out, is_raw = False)

# 這是 Python 腳本的標準入口判斷。
# 當這個檔案被直接執行時 (python main.py)，__name__ 會是 "__main__"，因此會執行 main()。
# 如果這個檔案是被其他程式 import，則不會自動執行 main()。
if __name__ == "__main__":
    main()