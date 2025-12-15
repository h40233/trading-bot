# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案是一個工具模組 (Utility Module)，主要用於處理數據與交易所 API 的互動。
# -----------------------------------------------------------------------------------------

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
from func_timeout import func_timeout, FunctionTimedOut

# 設定 Logging 的基礎配置。
logging.basicConfig(level=logging.INFO)

socket.setdefaulttimeout(20)

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def to_timestamp(time_str: str) -> int:
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    tz = timezone(timedelta(hours=8))
    ts = int(dt.replace(tzinfo=tz).timestamp()) * 1000
    return ts

def ms_to_str(ts):
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone(timedelta(hours=8)))
    return dt.strftime("%Y-%m-%d %H:%M")

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
    start_str = ms_to_str(start_time)
    end_str = ms_to_str(end_time)
    
    file_start_str = pd.to_datetime(start_time, unit="ms", utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    file_end_str = pd.to_datetime(end_time, unit="ms", utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    
    pathdir = "data/raw/"
    filename = f"{symbol}_{timeframe}_{file_start_str} to {file_end_str}.csv"
    path = f"{pathdir}/{filename}"
    logging.info(f"目標檔案路徑: {path}")
    
    if os.path.lexists(path):
        logging.info("已抓取過K線資料，直接調用")
        df = pd.read_csv(path)
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, False
    else:
        logging.info("尚未抓取過此段K線資料，開始抓取...")
        
        if client is None:
             raise ValueError("錯誤：API Key 未設定或連線失敗，無法下載資料。請檢查 .env 檔案。")

        df = []
        next_start_time = start_time
        
        retry_count = 0
        max_retries = 20 
        FORCE_TIMEOUT_SECONDS = 15
        SKIP_STEP_MS = 60 * 60 * 1000 
        
        while True:
            # --- [修正重點] 將結束檢查移到迴圈最開頭 ---
            # 這樣就算 except 區塊執行了 continue，回來這裡也會馬上被攔截
            if next_start_time >= end_time:
                print(f"\n[完成] 抓取時間已達設定結束點 ({ms_to_str(end_time)})")
                break
            # ----------------------------------------

            try:
                current_time_str = ms_to_str(next_start_time)
                total_duration = end_time - start_time
                current_duration = next_start_time - start_time
                # 避免計算超過 100%
                process = min(current_duration / total_duration, 1.0)
                
                print(f"[{process:.1%}] 正在請求: {current_time_str} ...", end="\r", flush=True)

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

                kline = func_timeout(FORCE_TIMEOUT_SECONDS, api_call)
                
                if not kline:
                    if next_start_time < end_time:
                         print(f"\n[提示] 該時段無資料，嘗試往後搜尋...")
                         next_start_time += SKIP_STEP_MS
                         continue
                    else:
                        print(f"\n[提示] 已無更多資料，停止抓取。")
                        break
                
                df.extend(kline)
                
                last_open_time = kline[-1][0]
                
                if last_open_time <= next_start_time:
                    next_start_time += 60000 
                else:
                    next_start_time = last_open_time + 1
                
                retry_count = 0 
                time.sleep(rate_limit)
            
            except (FunctionTimedOut, Exception) as e:
                retry_count += 1
                error_msg = str(e)
                
                if isinstance(e, FunctionTimedOut):
                    error_type = "強制超時"
                elif "502" in error_msg or "500" in error_msg:
                    error_type = "伺服器錯誤 (5xx)"
                else:
                    error_type = "連線異常"

                print(f"\n⚠️ [{error_type}] 第 {retry_count} 次重試...")
                
                if retry_count >= 5:
                    print(f"⏭️  異常頻繁，跳過 {SKIP_STEP_MS/1000/60} 分鐘...")
                    next_start_time += SKIP_STEP_MS
                    retry_count = 0 
                    time.sleep(2)
                    continue
                
                if retry_count > max_retries:
                    print("❌ 放棄。")
                    break
                
                wait_time = min(retry_count * 2, 20)
                time.sleep(wait_time)
        
        print("\n抓取流程結束，開始整理資料...")    

        if not df:
            # 如果真的沒抓到任何資料，但我們不想報錯中斷 (可能只是這段時間剛好沒行情)
            # 我們回傳一個空的 DataFrame，讓後面策略層去處理 (通常策略算不出指標就會回傳空結果)
            logging.warning("注意：本次未抓取到任何資料。")
            columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
            return pd.DataFrame(columns=columns), False

        columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
        df_result = pd.DataFrame(df, columns=columns)
        df_result.drop(columns=["ignore"], inplace=True)
        
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
        
        df_result["open_time"] = pd.to_datetime(df_result["open_time"], unit="ms", utc=True)
        df_result["close_time"] = pd.to_datetime(df_result["close_time"], unit="ms", utc=True)
        
        df_result["open_time"] = df_result["open_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df_result["close_time"] = df_result["close_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        
        df_result["symbol"] = symbol
        df_result.drop_duplicates(subset=['open_time'], inplace=True)
        
        return df_result, True

# 以下函式保持不變
def data_to_csv(df, is_raw, timeframe=None):
    if df.empty:
        logging.warning("資料為空，跳過存檔")
        return

    if timeframe is None:
        try:
            config = load_config()
            timeframe = config['基本設定']['timeframe']
        except:
            timeframe = "unknown"

    if is_raw:
        pathdir = "data/raw"
        filename = f"{df.loc[0,'symbol']}_{timeframe}_{df.loc[0,'open_time']} to {df.iloc[-1]['open_time']}"
    else:
        pathdir = "data/processed"
        filename = f"{df.loc[0,'symbol']}_{timeframe}_{df.loc[0, 'strategy_name']}_{df.loc[0,'open_time']} to {df.iloc[-1]['open_time']}"
    
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)

def to_csv(df, pathdir , filename):
    os.makedirs(pathdir, exist_ok=True)
    path = f"{pathdir}/{filename}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已儲存檔案到{pathdir}/")

def result_to_csv(df, is_backtest):
    config = load_config()
    if is_backtest:
        pathdir = "result/backtests"
    else:
        pathdir = "result/logs"
    filename = f"{config['基本設定']['symbol']}_{config['基本設定']['timeframe']}_{config['基本設定']['strategy']}_{df.loc[0,'時間']} to {df.iloc[-1]['時間']}"
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)

def load_strategy(strategy_name:str):
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"找不到策略 {strategy_name}")
    strategy = STRATEGY_REGISTRY[strategy_name]
    s = strategy()
    return s

def get_processed_data(filename:str):
        pathdir = "data/processed/"
        path = pathdir + filename
        if os.path.lexists(path):
            logging.info("正在調用策略訊號")
            return pd.read_csv(path)
        else:
            raise ValueError(f"{filename}不存在，請先使用策略取得訊號資料")