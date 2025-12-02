# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是資料載入模組 (Data Loader)。
# 它的功能是封裝「抓取資料 -> 計算策略 -> 儲存檔案」的完整流程。
# 它可以被獨立執行，也可以被 UI (Streamlit) 或其他程式呼叫。
# -----------------------------------------------------------------------------------------

import logging
import os
import sys
from typing import Optional, Dict, Any

# 引入 Binance SDK
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures, 
    ConfigurationRestAPI, 
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
)

# 引入專案工具
from util import load_config, to_timestamp, get_kline_data, data_to_csv, load_strategy

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_process_data(custom_config: Optional[Dict[str, Any]] = None):
    """
    執行完整的資料更新流程。
    
    參數:
        custom_config (dict, optional): 如果有傳入，將使用此設定覆蓋預設的 config.json。
                                        這對於 UI 修改參數後立即執行非常有用。
    
    回傳:
        DataFrame: 處理完成且包含策略訊號的資料表。如果不成功則回傳 None。
    """
    
    # 1. 決定使用哪一份設定 (傳入的 or 檔案讀取的)
    if custom_config:
        config = custom_config
        logging.info("使用外部傳入的設定參數進行更新")
    else:
        config = load_config()
        logging.info("使用 config.json 設定檔進行更新")

    try:
        # 2. 解析設定參數
        base_conf = config["基本設定"]
        backtest_conf = config["回測設定"]
        
        symbol = base_conf["symbol"]
        timeframe = base_conf["timeframe"]
        strategy_name = base_conf["strategy"]
        
        # 轉換時間戳記
        start_time = to_timestamp(backtest_conf["start_time"])
        end_time = to_timestamp(backtest_conf["end_time"])

        logging.info(f"準備更新資料: {symbol} [{timeframe}] | 策略: {strategy_name}")

        # 3. 建立 Binance Client 連線
        # 從環境變數讀取 Key (安全性考量)
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        
        # 判斷網址 (測試網 vs 正式網)
        url = base_conf["testnet_url"] if base_conf["testnet"] else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
        
        client = None
        # 如果有 API Key，則初始化 client；如果沒有，嘗試依賴本地快取 (util.py 會處理)
        if api_key:
            config_api = ConfigurationRestAPI(api_key=api_key, api_secret=api_secret, base_path=url)
            client = DerivativesTradingUsdsFutures(config_rest_api=config_api).rest_api
            logging.info("Binance Client 初始化成功")
        else:
            logging.warning("未偵測到 API KEY，將嘗試讀取本地快取資料 (若無快取將會失敗)")

        # 4. 抓取 K 線資料 (Raw Data)
        # util.get_kline_data 內部有快取機制，若檔案存在會直接讀檔
        df, needsave = get_kline_data(
            client=client,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            mark_price=base_conf["use_mark_price_kline"],
            limit=base_conf["fetch_limit"],
            rate_limit=base_conf["sleep_time"]
        )
        
        # 如果是新抓下來的資料，存入 data/raw
        if needsave:
            data_to_csv(df, is_raw=True)
            logging.info("原始資料已更新並存檔")

        # 5. 載入並執行策略
        logging.info(f"載入策略: {strategy_name}")
        strategy = load_strategy(strategy_name)
        
        logging.info("正在計算策略訊號...")
        df_processed = strategy.generate_signal(df)
        
        # 6. 儲存處理後的資料 (Processed Data)
        data_to_csv(df_processed, is_raw=False)
        logging.info("策略訊號計算完成，資料已存檔")
        
        return df_processed

    except Exception as e:
        logging.error(f"資料更新流程發生錯誤: {e}", exc_info=True)
        return None

# -----------------------------------------------------------------------------------------
# [程式入口]
# 當直接執行此檔案 (python data_loader.py) 時，會執行以下區塊
# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 執行更新
    result = fetch_and_process_data()
    
    if result is not None:
        print("\n" + "="*50)
        print("✅ 資料更新與策略計算成功！")
        print(f"資料長度: {len(result)} 筆")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("❌ 更新失敗，請檢查 Log 訊息")
        print("="*50 + "\n")