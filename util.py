from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFuturesRestAPI
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
import re
import time
import json
from strategies import STRATEGY_REGISTRY
import tqdm

logging.basicConfig(level=logging.INFO)

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        logging.info("載入設定檔成功")
        return json.load(f)

def to_timestamp(time : str)->int:
    """
    把時間格式字串轉成unix timestamp(毫秒 ms) \n
    格式僅限 "%Y-%m-%d %H:%M"
    """
    dt = datetime.strptime(time,"%Y-%m-%d %H:%M")
    tz = timezone(timedelta(hours = 8))
    ts = int(dt.replace(tzinfo=tz).timestamp())*1000
    return ts

def get_kline_data(client :DerivativesTradingUsdsFuturesRestAPI,
                    symbol,
                    timeframe,
                    start_time:int = None,
                    end_time:int = None,
                    mark_price:bool = False,
                    limit:int = None
                    ):
    """抓K線資料回傳pandas dataframe"""
    config = load_config()
    start_datetime = pd.to_datetime(start_time,unit="ms",utc=True)
    end_datetime = pd.to_datetime(end_time,unit="ms",utc=True)
    start_datetime = start_datetime.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    end_datetime = end_datetime.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    pathdir = "data/raw/"
    filename = f"{symbol}_{start_datetime} to {end_datetime}.csv"
    path = f"{pathdir}/{filename}"
    logging.info(path)
    if os.path.lexists(path):
        logging.info("已抓取過K線資料，直接調用")
        needsave = False
        return pd.read_csv(path), needsave
    else:
        logging.info("尚未抓取過此段K線資料，開始抓取")
        df = []
        count = 0
        next_start_time = start_time
        total = end_time - start_time
        pbar = tqdm(total, desc = "抓取進度")
        while True:
            try:
                if mark_price:
                    kline = client.mark_price_kline_candlestick_data(
                                                            symbol = symbol,
                                                            interval = timeframe,
                                                            start_time = next_start_time,
                                                            end_time = end_time,
                                                            limit = limit
                                                            ).data()
                else:
                    kline = client.kline_candlestick_data(
                                                            symbol = symbol,
                                                            interval = timeframe,
                                                            start_time = next_start_time,
                                                            end_time = end_time,
                                                            limit = limit
                                                            ).data()
                df.extend(kline)
                next_start_time = df[-1][0]+1
                if next_start_time >= end_time:
                    break
                
                count += 1
                logging.info(f"start_time = {pd.to_datetime(next_start_time, unit="ms",utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")}\nend_time = {pd.to_datetime(end_time, unit="ms",utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")}\n已迴圈{count}次")
                pbar.update(next_start_time - start_time - pbar.n)
                time.sleep(config["sleep_time"])
            except Exception as e:
                raise f"出現錯誤 {e}"
        
        pbar.close()

        columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
        df = pd.DataFrame(df, columns = columns)
        df.drop(columns=["ignore"],inplace=True)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["open_time"] = df["open_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df["close_time"] = df["close_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df["symbol"] = symbol
        needsave = True
        return df, needsave

def to_csv(df, is_raw):
    if is_raw:
        pathdir = "data/raw"
        filename = f"{df.loc[0,"symbol"]}_{df.loc[0,"open_time"]} to {df.iloc[-1]["open_time"]}"
    else:
        pathdir = "data/processed"
        filename = f"{df.loc[0,"symbol"]}_{df.loc[0, "strategy"]}_{df.loc[0,"open_time"]} to {df.iloc[-1]["open_time"]}"
    filename = re.sub(":", "-", filename)

    os.makedirs(pathdir, exist_ok=True)
    path = f"{pathdir}/{filename}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已儲存檔案到{pathdir}/")

def load_strategy(strategy_name:str):
    #載入策略
    logging.info(STRATEGY_REGISTRY)
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError("找不到策略{strategy_name}")
    strategy = STRATEGY_REGISTRY[strategy_name]
    return strategy