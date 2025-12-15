# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是資料載入模組 (Data Loader)。
# 它的功能是封裝「抓取資料 -> 計算策略 -> 儲存檔案」的完整流程。
# -----------------------------------------------------------------------------------------

import logging
import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 引入 Binance SDK (只保留 PROD URL，因為 Testnet 我們要用 config 的)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures, 
    ConfigurationRestAPI, 
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
)

# 引入專案工具
from util import load_config, to_timestamp, get_kline_data, data_to_csv, load_strategy

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_process_data(custom_config: Optional[Dict[str, Any]] = None):
    """
    執行完整的資料更新流程。
    """
    
    # 1. 決定使用哪一份設定
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
        
        start_time = to_timestamp(backtest_conf["start_time"])
        end_time = to_timestamp(backtest_conf["end_time"])

        logging.info(f"準備更新資料: {symbol} [{timeframe}] | 策略: {strategy_name}")

        # 3. 建立 Binance Client 連線
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        
        # [修正] 改回使用 config 裡的 testnet_url
        url = base_conf["testnet_url"] if base_conf["testnet"] else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
        
        client = None
        if api_key:
            config_api = ConfigurationRestAPI(api_key=api_key, api_secret=api_secret, base_path=url)
            client = DerivativesTradingUsdsFutures(config_rest_api=config_api).rest_api
            logging.info("Binance Client 初始化成功")
        else:
            logging.warning("未偵測到 API KEY，將嘗試讀取本地快取資料 (若無快取將會失敗)")

        # 4. 抓取 K 線資料 (Raw Data)
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
        
        # 呼叫 data_to_csv 時傳入 timeframe
        if needsave:
            data_to_csv(df, is_raw=True, timeframe=timeframe)
            logging.info("原始資料已更新並存檔")

        # 5. 載入並執行策略
        logging.info(f"載入策略: {strategy_name}")
        strategy = load_strategy(strategy_name)
        
        logging.info("正在計算策略訊號...")
        df_processed = strategy.generate_signal(df)
        
        # 呼叫 data_to_csv 時傳入 timeframe
        data_to_csv(df_processed, is_raw=False, timeframe=timeframe)
        logging.info("策略訊號計算完成，資料已存檔")
        
        return df_processed

    except Exception as e:
        logging.error(f"資料更新流程發生錯誤: {e}", exc_info=True)
        return None

if __name__ == "__main__":
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