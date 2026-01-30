# =========================================================================================
# 檔案名稱：data_loader.py
# 檔案說明：
#   本檔案實作了「資料載入模組 (Data Loader)」。
#   它是一個高階的 Facade (外觀模式)，封裝了底層的 API 呼叫與資料處理細節。
#   使用者或 UI 只需呼叫 fetch_and_process_data() 即可完成從「抓 K 線」到「算訊號」的全套流程。
#
# 流程步驟：
#   1. 設定讀取：從 config.json 或參數載入設定。
#   2. API 連線：初始化 Binance 客戶端。
#   3. 資料獲取：呼叫 util.get_kline_data 下載或讀取快取。
#   4. 策略計算：動態載入策略並計算進出場訊號。
#   5. 存檔：將處理好的資料 (Processed Data) 存為 CSV。
# =========================================================================================

# [模組引用解析]
import logging
import os
import sys
# 引入型別提示，增加程式碼可讀性。
from typing import Optional, Dict, Any
# 引入 dotenv 以讀取 .env 檔案中的敏感資訊 (如 API Key)。
from dotenv import load_dotenv

# 引入 Binance SDK 相關類別 (負責與交易所溝通)。
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures, 
    ConfigurationRestAPI, 
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
)

# 引入專案內部的工具函式庫。
from util import load_config, to_timestamp, get_kline_data, data_to_csv, load_strategy

# 載入環境變數 (通常用於開發環境)。
load_dotenv()

# 設定日誌格式：時間 - 等級 - 訊息。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# [函式說明]
# 功能：執行完整的資料更新流程。
# 參數：custom_config (選填)，若有傳入則覆蓋預設的 config.json (常用於優化模式)。
def fetch_and_process_data(custom_config: Optional[Dict[str, Any]] = None):
    """
    執行完整的資料更新流程。
    """
    
    # 1. 決定使用哪一份設定
    # [控制流解析] 設定優先級
    # 若外部有傳入 custom_config (例如從 UI 修改了參數)，優先使用它；否則讀取磁碟上的設定檔。
    if custom_config:
        config = custom_config
        logging.info("使用外部傳入的設定參數進行更新")
    else:
        config = load_config()
        logging.info("使用 config.json 設定檔進行更新")

    try:
        # 2. 解析設定參數
        # 將設定檔中的巢狀結構展開，方便後續取用。
        base_conf = config["基本設定"]
        backtest_conf = config["回測設定"]
        
        symbol = base_conf["symbol"]
        timeframe = base_conf["timeframe"]
        strategy_name = base_conf["strategy"]
        
        # 將日期字串 ("2023-01-01") 轉換為 Unix Timestamp (毫秒)。
        start_time = to_timestamp(backtest_conf["start_time"])
        end_time = to_timestamp(backtest_conf["end_time"])

        logging.info(f"準備更新資料: {symbol} [{timeframe}] | 策略: {strategy_name}")

        # 3. 建立 Binance Client 連線
        # 從環境變數中讀取 API 金鑰 (安全性考量，不應硬編碼在程式中)。
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        
        # [邏輯] 網域選擇
        # 若設定為測試網 (Testnet)，則使用 config 中的 URL；否則使用正式環境 URL。
        url = base_conf["testnet_url"] if base_conf["testnet"] else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
        
        client = None
        if api_key:
            # 初始化 REST API 客戶端
            config_api = ConfigurationRestAPI(api_key=api_key, api_secret=api_secret, base_path=url)
            client = DerivativesTradingUsdsFutures(config_rest_api=config_api).rest_api
            logging.info("Binance Client 初始化成功")
        else:
            logging.warning("未偵測到 API KEY，將嘗試讀取本地快取資料 (若無快取將會失敗)")

        # 4. 抓取 K 線資料 (Raw Data)
        # 呼叫 util.get_kline_data，該函式內部實作了「快取優先」與「斷點續傳」邏輯。
        # 回傳：df (資料表), needsave (布林值，指示是否為新下載的資料)。
        df, needsave = get_kline_data(
            client=client,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            mark_price=base_conf["use_mark_price_kline"], # 是否使用標記價格
            limit=base_conf["fetch_limit"],               # 單次 API 請求的 K 線數量
            rate_limit=base_conf["sleep_time"]            # API 請求間隔 (防止 Ban IP)
        )
        
        # 若有新下載資料，則存檔至 data/raw/。
        if needsave:
            data_to_csv(df, is_raw=True, timeframe=timeframe)
            logging.info("原始資料已更新並存檔")

        # 5. 載入並執行策略
        # 透過反射機制 (Reflection) 動態實例化策略物件。
        logging.info(f"載入策略: {strategy_name}")
        strategy = load_strategy(strategy_name)
        
        logging.info("正在計算策略訊號...")
        # 將原始 K 線傳入策略，獲得帶有 signal 欄位的 DataFrame。
        df_processed = strategy.generate_signal(df)
        
        # 將計算結果存檔至 data/processed/，供回測引擎使用。
        data_to_csv(df_processed, is_raw=False, timeframe=timeframe)
        logging.info("策略訊號計算完成，資料已存檔")
        
        return df_processed

    except Exception as e:
        # 捕捉流程中任何未預期的錯誤，並記錄堆疊追蹤 (Stack Trace)。
        logging.error(f"資料更新流程發生錯誤: {e}", exc_info=True)
        return None

# [程式進入點]
# 說明：單獨執行此檔案時，會執行預設的資料更新流程。
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