from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFuturesRestAPI
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
import re
import time
import json
from strategies import STRATEGY_REGISTRY

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
                    limit:int = None,
                    rate_limit:float = 0.3
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
        needsave = False
        return pd.read_csv(path), needsave
    else:
        logging.info("尚未抓取過此段K線資料，開始抓取")
        df = []
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
                process = (next_start_time - start_time) / (end_time - start_time)
                print(f"抓取進度{process:.2%}",end="\r",flush=True)
                next_start_time = df[-1][0]+1
                if next_start_time >= end_time:
                    break
                time.sleep(rate_limit)
            except Exception as e:
                raise f"發生錯誤{e}"
        print("抓取完成")    

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

def data_to_csv(df, is_raw):
    if is_raw:
        pathdir = "data/raw"
        filename = f"{df.loc[0,"symbol"]}_{df.loc[0,"open_time"]} to {df.iloc[-1]["open_time"]}"
    else:
        pathdir = "data/processed"
        filename = f"{df.loc[0,"symbol"]}_{df.loc[0, "strategy_name"]}_{df.loc[0,"open_time"]} to {df.iloc[-1]["open_time"]}"
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)

def to_csv(df, pathdir , filename):
    os.makedirs(pathdir, exist_ok=True)
    path = f"{pathdir}/{filename}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已儲存檔案到{pathdir}/")

def result_to_csv(df, is_backtest):
    if is_backtest:
        pathdir = "result/backtests"
    else:
        pathdir = "result/logs"
    filename = f"{df.loc[0,"symbol"]}_{df.loc[0, "strategy_name"]}_{df.loc[0,"open_time"]} to {df.iloc[-1]["open_time"]}"
    filename = re.sub(":", "-", filename)
    to_csv(df, pathdir, filename)

def load_strategy(strategy_name:str):
    #載入策略
    logging.info(STRATEGY_REGISTRY)
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError("找不到策略{strategy_name}")
    strategy = STRATEGY_REGISTRY[strategy_name]
    s = strategy()
    logging.info(f"載入策略{s.name}成功")
    return s

def get_processed_data(filename:str):
        pathdir = "data/processed/"
        path = pathdir + filename
        if os.path.lexists(path):
            logging.info("正在調用策略訊號")
            return pd.read_csv(path)
        else:
            raise ValueError(f"{filename}不存在，請先使用策略取得訊號資料")