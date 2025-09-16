from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (DerivativesTradingUsdsFutures,ConfigurationRestAPI,DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,DerivativesTradingUsdsFuturesRestAPI)
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
import re
import time

logging.basicConfig(level=logging.INFO)

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
        return pd.read_csv(path)
    else:
        logging.info("尚未抓取過此段K線資料，開始抓取")
        df = []
        count = 0
        next_start_time = start_time
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
                next_start_time = df[-1][0]
                if next_start_time >= end_time:
                    break
                
                count += 1
                logging.info(f"start_time = {pd.to_datetime(next_start_time, unit="ms",utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")}\nend_time = {pd.to_datetime(end_time, unit="ms",utc=True).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")}\n已迴圈{count}次")
                time.sleep(1)
            except Exception as e:
                raise f"出現錯誤 {e}"
        
        columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
        df = pd.DataFrame(df, columns = columns)
        df.drop(columns=["ignore"],inplace=True)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["open_time"] = df["open_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df["close_time"] = df["close_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df["symbol"] = symbol
        return df

def to_csv(df, pathdir:str, filename:str):
    os.makedirs("data/raw", exist_ok=True)
    path = f"{pathdir}/{filename}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已儲存檔案到{pathdir}/")

def kline_to_csv(df):
    pathdir = "data/raw"
    filename = f"{df.loc[0,"symbol"]}_{df.loc[0,"open_time"]} to {df.iloc[-1]["open_time"]}"
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)