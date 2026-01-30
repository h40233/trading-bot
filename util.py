# =========================================================================================
# 檔案名稱：util.py
# 檔案說明：
#   本檔案是「通用工具庫 (Utility Module)」。
#   集合了專案中重複使用的底層功能，包括設定檔讀取、時間格式轉換、API 資料抓取與檔案 I/O。
#
# 主要功能：
#   1. get_kline_data: 具備「快取檢查」、「分頁抓取」與「自動重試」功能的 K 線下載器。
#   2. load_config: 讀取 JSON 設定檔。
#   3. to_timestamp/ms_to_str: 時間格式轉換工具。
# =========================================================================================

# [模組引用解析]
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFuturesRestAPI
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
import re
import time
import json
import socket 
from strategies import STRATEGY_REGISTRY
# 引入 func_timeout 用於強制中止卡死的 API 請求。
from func_timeout import func_timeout, FunctionTimedOut

# 設定 Logging 基礎配置。
logging.basicConfig(level=logging.INFO)

# 設定 Socket 全域超時時間，防止網路連線無限期等待。
socket.setdefaulttimeout(20)

# [函式：讀取設定]
def load_config():
    """讀取並解析 config.json 檔案"""
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

# [函式：時間轉換 (字串 -> 毫秒)]
def to_timestamp(time_str: str) -> int:
    """將 'YYYY-MM-DD HH:MM' 字串轉換為 UTC+8 的毫秒 Timestamp"""
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    # 設定時區為 UTC+8 (台北時間)
    tz = timezone(timedelta(hours=8))
    # 轉換為 Timestamp 並乘 1000 變為毫秒
    ts = int(dt.replace(tzinfo=tz).timestamp()) * 1000
    return ts

# [函式：時間轉換 (毫秒 -> 字串)]
def ms_to_str(ts):
    """將毫秒 Timestamp 轉換為 'YYYY-MM-DD HH:MM' 字串 (UTC+8)"""
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone(timedelta(hours=8)))
    return dt.strftime("%Y-%m-%d %H:%M")

# [函式：獲取 K 線資料 (核心功能)]
# 邏輯：
#   1. 檢查本地是否已有對應時間段的 CSV 檔案 (快取機制)。
#   2. 若無，則啟動分頁抓取迴圈 (While Loop)。
#   3. 使用 func_timeout 包裹 API 呼叫，防止請求卡死。
#   4. 實作指數退避 (Exponential Backoff) 的重試機制。
def get_kline_data(client: DerivativesTradingUsdsFuturesRestAPI,
                   symbol,
                   timeframe,
                   start_time: int = None,
                   end_time: int = None,
                   mark_price: bool = False,
                   limit: int = None,
                   rate_limit: float = 0.5 
                   ):
    """抓K線資料回傳pandas dataframe"""
    
    # 格式化檔名所需的時間字串 (用於檢查快取)
    # 使用 UTC 時間轉 Asia/Taipei，避免跨日問題。
    start_str = ms_to_str(start_time)
    end_str = ms_to_str(end_time)
    
    file_start_str = pd.to_datetime(start_time, unit="ms", utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    file_end_str = pd.to_datetime(end_time, unit="ms", utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    
    pathdir = "data/raw/"
    filename = f"{symbol}_{timeframe}_{file_start_str} to {file_end_str}.csv"
    path = f"{pathdir}/{filename}"
    logging.info(f"目標檔案路徑: {path}")
    
    # [快取檢查]
    # 若檔案存在，直接讀取並回傳，不呼叫 API。
    if os.path.lexists(path):
        logging.info("已抓取過K線資料，直接調用")
        df = pd.read_csv(path)
        # 強制轉換數值欄位，避免字串混入。
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, False # False 代表「不需要存檔」
    else:
        logging.info("尚未抓取過此段K線資料，開始抓取...")
        
        # 若無快取且未提供 Client，則無法下載，拋出錯誤。
        if client is None:
             raise ValueError("錯誤：API Key 未設定或連線失敗，無法下載資料。請檢查 .env 檔案。")

        df = []
        next_start_time = start_time
        
        retry_count = 0
        max_retries = 20 # 最大重試次數
        FORCE_TIMEOUT_SECONDS = 15 # API 強制超時秒數
        SKIP_STEP_MS = 60 * 60 * 1000 # 若遇到壞檔區間，跳過的步長 (1小時)
        
        # [分頁抓取迴圈]
        while True:
            # [修正重點] 迴圈終止條件檢查
            # 必須放在迴圈開頭，確保 continue 後也能被檢查到。
            if next_start_time >= end_time:
                print(f"\n[完成] 抓取時間已達設定結束點 ({ms_to_str(end_time)})")
                break

            try:
                # 計算並顯示進度百分比
                current_time_str = ms_to_str(next_start_time)
                total_duration = end_time - start_time
                current_duration = next_start_time - start_time
                process = min(current_duration / total_duration, 1.0)
                
                print(f"[{process:.1%}] 正在請求: {current_time_str} ...", end="\r", flush=True)

                # 定義 API 呼叫的 Lambda 函式 (區分標記價格或一般價格)
                if mark_price:
                    api_call = lambda: client.mark_price_kline_candlestick_data(
                                symbol=symbol, interval=timeframe,
                                start_time=next_start_time, end_time=end_time, limit=limit
                            ).data()
                else:
                    api_call = lambda: client.kline_candlestick_data(
                                symbol=symbol, interval=timeframe,
                                start_time=next_start_time, end_time=end_time, limit=limit
                            ).data()

                # [關鍵保護] 使用 func_timeout 強制執行時間限制
                # 避免 requests 套件在網路不穩時無限卡住 (Hang)。
                kline = func_timeout(FORCE_TIMEOUT_SECONDS, api_call)
                
                # 若 API 回傳空列表，代表該時段無交易數據。
                if not kline:
                    if next_start_time < end_time:
                         print(f"\n[提示] 該時段無資料，嘗試往後搜尋...")
                         next_start_time += SKIP_STEP_MS # 跳過一段時間再試
                         continue
                    else:
                        print(f"\n[提示] 已無更多資料，停止抓取。")
                        break
                
                # 將新抓到的資料追加到列表中
                df.extend(kline)
                
                # 更新下一次抓取的起始時間
                last_open_time = kline[-1][0]
                if last_open_time <= next_start_time:
                    next_start_time += 60000 # 防呆：若時間沒推進，強制加 1 分鐘
                else:
                    next_start_time = last_open_time + 1
                
                # 成功獲取後，重置重試計數器。
                retry_count = 0 
                time.sleep(rate_limit) # 遵守 API Rate Limit
            
            except (FunctionTimedOut, Exception) as e:
                # [錯誤處理與重試機制]
                retry_count += 1
                error_msg = str(e)
                
                # 判斷錯誤類型以顯示更友善的訊息
                if isinstance(e, FunctionTimedOut):
                    error_type = "強制超時"
                elif "502" in error_msg or "500" in error_msg:
                    error_type = "伺服器錯誤 (5xx)"
                else:
                    error_type = "連線異常"

                print(f"\n⚠️ [{error_type}] 第 {retry_count} 次重試...")
                
                # 若連續失敗次數過多，嘗試跳過該區間 (避免卡在某個壞點)。
                if retry_count >= 5:
                    print(f"⏭️  異常頻繁，跳過 {SKIP_STEP_MS/1000/60} 分鐘...")
                    next_start_time += SKIP_STEP_MS
                    retry_count = 0 
                    time.sleep(2)
                    continue
                
                # 若超過最大重試次數，宣告放棄。
                if retry_count > max_retries:
                    print("❌ 放棄。")
                    break
                
                # 指數退避等待 (wait_time 隨次數增加)。
                wait_time = min(retry_count * 2, 20)
                time.sleep(wait_time)
        
        print("\n抓取流程結束，開始整理資料...")    

        # 處理空資料情況
        if not df:
            logging.warning("注意：本次未抓取到任何資料。")
            columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
            return pd.DataFrame(columns=columns), False

        # [資料整理] 將 Raw List 轉為 DataFrame
        columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
        df_result = pd.DataFrame(df, columns=columns)
        # 移除不需要的欄位 (Volume, Quote Volume 等若不需要可在此處篩選)
        df_result.drop(columns=["ignore"], inplace=True)
        
        # 轉換數值型態
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
        
        # 處理時間格式
        df_result["open_time"] = pd.to_datetime(df_result["open_time"], unit="ms", utc=True)
        df_result["close_time"] = pd.to_datetime(df_result["close_time"], unit="ms", utc=True)
        
        # 轉為台北時間字串
        df_result["open_time"] = df_result["open_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df_result["close_time"] = df_result["close_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        
        df_result["symbol"] = symbol
        # 去除重複資料 (以 open_time 為鍵)
        df_result.drop_duplicates(subset=['open_time'], inplace=True)
        
        return df_result, True # True 代表「需要存檔」

# [函式：CSV 存檔]
def data_to_csv(df, is_raw, timeframe=None):
    """將 DataFrame 儲存為 CSV"""
    if df.empty:
        logging.warning("資料為空，跳過存檔")
        return

    # 若未指定 timeframe，嘗試從 config 讀取
    if timeframe is None:
        try:
            config = load_config()
            timeframe = config['基本設定']['timeframe']
        except:
            timeframe = "unknown"

    # 決定儲存路徑與檔名
    if is_raw:
        pathdir = "data/raw"
        filename = f"{df.loc[0,'symbol']}_{timeframe}_{df.loc[0,'open_time']} to {df.iloc[-1]['open_time']}"
    else:
        pathdir = "data/processed"
        filename = f"{df.loc[0,'symbol']}_{timeframe}_{df.loc[0, 'strategy_name']}_{df.loc[0,'open_time']} to {df.iloc[-1]['open_time']}"
    
    # 替換檔名中不合法的字元 (:)
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)

# [內部函式：實際執行存檔]
def to_csv(df, pathdir , filename):
    os.makedirs(pathdir, exist_ok=True)
    path = f"{pathdir}/{filename}.csv"
    # 使用 utf-8-sig 編碼以支援 Excel 開啟中文不亂碼
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已儲存檔案到{pathdir}/")

# [函式：回測結果存檔]
def result_to_csv(df, is_backtest):
    config = load_config()
    if is_backtest:
        pathdir = "result/backtests"
    else:
        pathdir = "result/logs"
    filename = f"{config['基本設定']['symbol']}_{config['基本設定']['timeframe']}_{config['基本設定']['strategy']}_{df.loc[0,'時間']} to {df.iloc[-1]['時間']}"
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)

# [函式：動態載入策略]
def load_strategy(strategy_name:str):
    """根據策略名稱，從註冊表中實例化策略物件"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"找不到策略 {strategy_name}")
    strategy = STRATEGY_REGISTRY[strategy_name]
    s = strategy()
    return s

# [函式：讀取已處理資料]
def get_processed_data(filename:str):
    """讀取 data/processed/ 下的訊號檔"""
    pathdir = "data/processed/"
    path = pathdir + filename
    if os.path.lexists(path):
        logging.info("正在調用策略訊號")
        return pd.read_csv(path)
    else:
        raise ValueError(f"{filename}不存在，請先使用策略取得訊號資料")