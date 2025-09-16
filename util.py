from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (DerivativesTradingUsdsFutures,ConfigurationRestAPI,DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,DerivativesTradingUsdsFuturesRestAPI)
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
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
                    mark_price:bool = False
                    ):
    """抓K線資料回傳pandas dataframe"""
    if mark_price:
        kline = client.mark_price_kline_candlestick_data(
                                                symbol = symbol,
                                                interval = timeframe,
                                                start_time = start_time,
                                                end_time = end_time
                                                ).data()
    else:
        kline = client.kline_candlestick_data(
                                                symbol = symbol,
                                                interval = timeframe,
                                                start_time = start_time,
                                                end_time = end_time
                                                ).data()
        
    columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
    df = pd.DataFrame(kline, columns = columns)
    df.drop(columns=["ignore"],inplace=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["open_time"] = df["open_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
    df["close_time"] = df["close_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
    return df

def to_csv(df, pathdir:str, filename:str):
    os.makedirs("data/raw", exist_ok=True)
    path = f"{pathdir}/{filename}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已儲存檔案到{pathdir}")

def kline_to_csv(df):
    pathdir = "data/raw"
    filename = f"{df.loc[0,"open_time"]} to {df.iloc[-1]["close_time"]}"
    to_csv(df, pathdir, filename)