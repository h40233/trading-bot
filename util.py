# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案是一個工具模組 (Utility Module)，主要用於處理數據與交易所 API 的互動。
# 它的核心功能包含：
# 1. 讀取設定檔 (Config)。
# 2. 時間格式轉換 (Unix Timestamp 與 DateTime 互轉)。
# 3. 透過 Binance SDK 抓取 K線 (Candlestick) 歷史資料，並處理分頁請求 (Pagination)。
# 4. 將資料儲存為 CSV 檔案，或讀取已儲存的檔案 (快取機制)。
# 5. 動態載入交易策略 (Strategy Loading)。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 引入 Binance 的 Python SDK，這是專門用來跟必安 U本位合約 (USDS-M Futures) 互動的介面。
# 我們需要它來發送 API 請求，獲取市場數據。
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFuturesRestAPI

# 引入 os 模組 (Operating System)，用於與作業系統互動。
# 這裡主要用來檢查檔案是否存在 (os.path.exists) 以及建立資料夾 (os.makedirs)。
import os

# 引入 datetime 相關模組，用於處理時間。
# datetime: 處理日期與時間的主體。
# timezone: 設定時區 (如 UTC+8)。
# timedelta: 處理時間的加減運算 (如加 8 小時)。
from datetime import datetime, timezone, timedelta

# 引入 pandas，這是 Python 數據分析的神器。
# 我們用它來將抓回來的原始數據整理成表格 (DataFrame)，方便後續運算與存檔。
import pandas as pd

# 引入 logging，用於紀錄程式執行的過程 (Log)。
# 比起 print，logging 可以設定等級 (INFO, ERROR)，更適合正式專案開發。
import logging

# 引入 re (Regular Expression)，正規表達式。
# 這裡用來處理字串，例如把檔名中不合法的符號 (如冒號 :) 替換掉。
import re

# 引入 time 模組。
# 主要用到 time.sleep()，讓程式暫停一下，避免請求太快被交易所封鎖 (Rate Limit)。
import time

# 引入 json 模組。
# 用來讀取與解析 'config.json' 設定檔，將 JSON 格式轉為 Python 字典。
import json

# 從 strategies 資料夾引入 STRATEGY_REGISTRY。
# 這是一個字典，紀錄了所有可用的策略名稱與對應的 Class，讓我們能動態選擇要跑哪個策略。
from strategies import STRATEGY_REGISTRY

# 設定 Logging 的基礎配置。
# level=logging.INFO 表示「普通資訊」及更嚴重的訊息都會被印出來，方便我們監控程式狀態。
logging.basicConfig(level=logging.INFO)

# [Function 說明]
# 功能：讀取專案目錄下的 'config.json' 設定檔。
# 原理：使用 Python 的 open 函式讀取檔案，並用 json.load 解析成字典回傳。
def load_config():
    # 使用 'with open' 語法開啟檔案。
    # "r" 代表讀取 (read)，encoding="utf-8" 確保能正確讀取中文字符。
    # with 語法的好處是，當程式區塊結束後，它會自動關閉檔案，避免佔用資源。
    with open("config.json", "r", encoding="utf-8") as f:
        # 紀錄一條 Log，告知使用者設定檔讀取成功。
        logging.info("載入設定檔成功")
        # 將檔案內容 (JSON 字串) 轉換成 Python 的字典 (dict) 或列表 (list) 並回傳。
        return json.load(f)

# [Function 說明]
# 功能：將人類可讀的時間字串 (如 "2023-01-01 08:00") 轉換為 Unix Timestamp (毫秒)。
# 原理：Binance API 通常只接受毫秒格式的時間戳記，所以需要將字串轉為 datetime 物件，再轉為數字。
# 注意：這裡強制設定為 UTC+8 (台灣時間)。
def to_timestamp(time : str)->int:
    """
    把時間格式字串轉成unix timestamp(毫秒 ms) \n
    格式僅限 "%Y-%m-%d %H:%M"
    """
    # 將字串依照 "%Y-%m-%d %H:%M" 的格式解析成 datetime 物件。
    dt = datetime.strptime(time,"%Y-%m-%d %H:%M")
    # 建立一個時區物件，設定為 UTC+8 (台灣時間)。
    tz = timezone(timedelta(hours = 8))
    # 1. dt.replace(tzinfo=tz): 將原本沒有時區的時間，強制標記為 UTC+8。
    # 2. .timestamp(): 將時間轉換為 Unix Timestamp (秒，浮點數)。
    # 3. *1000: API 需要毫秒 (ms)，所以乘以 1000。
    # 4. int(): 轉成整數，去掉小數點。
    ts = int(dt.replace(tzinfo=tz).timestamp())*1000
    # 回傳計算好的毫秒時間戳記。
    return ts

# [Function 說明]
# 功能：向 Binance 抓取 K 線 (K-line/Candlestick) 資料，並整理成 Pandas DataFrame。
# 原理：
# 1. 檢查硬碟是否已經有這段時間的 CSV 檔，如果有就直接讀取 (快取機制)，節省時間。
# 2. 如果沒有，就使用 while 迴圈分批向 API 請求數據 (因為 API 一次通常只能回傳 1000 筆)。
# 3. 抓取完畢後，清洗數據格式，並存成 CSV。
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
    # 將傳入的開始時間戳記 (毫秒) 轉為 datetime 物件，並設定為 UTC 時間。
    start_datetime = pd.to_datetime(start_time,unit="ms",utc=True)
    # 將傳入的結束時間戳記 (毫秒) 轉為 datetime 物件，並設定為 UTC 時間。
    end_datetime = pd.to_datetime(end_time,unit="ms",utc=True)
    # 將 UTC 時間轉換為台灣時間 (Asia/Taipei)，並格式化成字串，用於建立檔名。
    start_datetime = start_datetime.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    # 同上，處理結束時間的格式。
    end_datetime = end_datetime.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H-%M")
    # 設定原始資料存放的資料夾路徑。
    pathdir = "data/raw/"
    # 組合檔名：商品名稱_開始時間 to 結束時間.csv。
    filename = f"{symbol}_{start_datetime} to {end_datetime}.csv"
    # 組合完整的檔案路徑 (資料夾 + 檔名)。
    path = f"{pathdir}/{filename}"
    # 印出檔案路徑確認。
    logging.info(path)
    
    # 檢查這個路徑的檔案是否存在。
    if os.path.lexists(path):
        # 如果存在，印出訊息，不重新下載。
        logging.info("已抓取過K線資料，直接調用")
        # 標記不需要存檔 (因為已經有了)。
        needsave = False
        # 直接讀取 CSV 並回傳 DataFrame 與 needsave 旗標。
        return pd.read_csv(path), needsave
    else:
        # 如果檔案不存在，準備開始下載流程。
        logging.info("尚未抓取過此段K線資料，開始抓取")
        # 初始化一個空列表，用來存放每次 API 回傳的一小段 K 線數據。
        df = []
        # 設定下一次請求的開始時間，初始值為這段區間的起點。
        next_start_time = start_time
        
        # 進入無窮迴圈，直到抓完所有資料為止。
        while True:
            try:
                # 判斷是否要抓取「標記價格 (Mark Price)」的 K 線。
                if mark_price:
                    # 使用 SDK 呼叫標記價格 K 線 API。
                    kline = client.mark_price_kline_candlestick_data(
                                                                    symbol = symbol,          # 交易對 (如 BTCUSDT)
                                                                    interval = timeframe,     # 時間週期 (如 1h)
                                                                    start_time = next_start_time, # 本次請求的開始時間
                                                                    end_time = end_time,      # 請求的結束時間
                                                                    limit = limit             # 限制筆數 (通常 API 預設 500 或 1000)
                                                                    ).data() # .data() 取出回傳內容
                else:
                    # 如果不是標記價格，則抓取一般交易價格的 K 線。
                    kline = client.kline_candlestick_data(
                                                                    symbol = symbol,
                                                                    interval = timeframe,
                                                                    start_time = next_start_time,
                                                                    end_time = end_time,
                                                                    limit = limit
                                                                    ).data()
                # 將這次抓到的一批數據 (kline) 加入到總列表 (df) 中。
                df.extend(kline)
                
                # 計算目前的抓取進度 (已過時間 / 總時間長度)，用於顯示進度條。
                process = (next_start_time - start_time) / (end_time - start_time)
                # print 進度，end="\r" 讓游標回到行首，flush=True 強制輸出，造成動態更新數字的效果。
                print(f"抓取進度{process:.2%}",end="\r",flush=True)
                
                # 更新下一次請求的開始時間。
                # df[-1][0] 是這批資料最後一筆的開盤時間，加 1 毫秒作為下一批的起點，避免重複抓取。
                next_start_time = df[-1][0]+1
                
                # 判斷條件：如果下一次開始時間已經超過或等於我們設定的結束時間，就停止迴圈。
                if next_start_time >= end_time:
                    break
                
                # 暫停一下 (rate_limit)，避免因為請求太頻繁被交易所封鎖 IP。
                time.sleep(rate_limit)
            
            # 錯誤處理機制。
            except Exception as e:
                # 如果發生錯誤 (如網路斷線)，記錄錯誤訊息。
                logging.error(f"抓取K線時發生錯誤: {e}")
                # 重新拋出 (raise) 這個錯誤，讓程式中斷或由上層處理，確保我們知道出錯了。
                raise # 重新拋出原始異常，保留堆疊信息
        
        # 迴圈結束，表示抓取完成。
        print("抓取完成")    

        # 定義 K 線資料的欄位名稱 (這是 Binance API 的標準回傳順序)。
        columns = ["open_time", "open", "high", "low", "close", "ignore", "close_time", "ignore","ignore","ignore","ignore","ignore"]
        # 將列表資料轉換成 Pandas DataFrame 格式。
        df = pd.DataFrame(df, columns = columns)
        # 刪除不需要的欄位 ("ignore" 欄位通常是成交額等我們暫時不用的數據)。
        df.drop(columns=["ignore"],inplace=True)
        
        # 將 open_time 欄位轉為 datetime 格式 (UTC)。
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        # 將 close_time 欄位轉為 datetime 格式 (UTC)。
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        
        # 將時間轉換為台灣時區 (Asia/Taipei) 並轉成字串格式，方便閱讀。
        df["open_time"] = df["open_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        df["close_time"] = df["close_time"].dt.tz_convert("Asia/Taipei").dt.strftime("%Y-%m-%d %H:%M")
        
        # 新增一個 symbol 欄位，標記這份資料屬於哪個幣種。
        df["symbol"] = symbol
        # 標記需要存檔 (因為是新抓下來的)。
        needsave = True
        # 回傳整理好的 DataFrame 和 needsave 旗標。
        return df, needsave

# [Function 說明]
# 功能：決定將 DataFrame 儲存為 CSV 的檔名與路徑。
# 原理：依據資料是「原始資料 (raw)」還是「處理後資料 (processed)」來決定存放位置和命名規則。
def data_to_csv(df, is_raw):
    # 如果是原始 K 線資料。
    if is_raw:
        # 設定路徑為 data/raw。
        pathdir = "data/raw"
        # 檔名格式：幣種_開始時間 to 結束時間。
        filename = f"{df.loc[0,'symbol']}_{df.loc[0,'open_time']} to {df.iloc[-1]['open_time']}"
    else:
        # 如果是策略處理過的資料。
        pathdir = "data/processed"
        # 檔名格式：幣種_策略名稱_開始時間 to 結束時間。
        filename = f"{df.loc[0,'symbol']}_{df.loc[0, 'strategy_name']}_{df.loc[0,'open_time']} to {df.iloc[-1]['open_time']}"
    
    # Windows 檔名不能包含冒號 (:)，所以用正規表達式把時間中的冒號換成連字號 (-)。
    filename = re.sub(":", "-", filename)
    # 呼叫底層的 to_csv 函式執行實際儲存動作。
    to_csv(df, pathdir, filename)

# [Function 說明]
# 功能：執行實際的 CSV 寫入動作。
# 原理：先確保資料夾存在，然後使用 Pandas 的 to_csv 方法存檔。
def to_csv(df, pathdir , filename):
    # 如果路徑中的資料夾不存在，就自動建立 (包括所有父資料夾)。
    os.makedirs(pathdir, exist_ok=True)
    # 組合完整的檔案路徑。
    path = f"{pathdir}/{filename}.csv"
    # 將 DataFrame 寫入 CSV。
    # index=False: 不存 Pandas 的索引欄 (0, 1, 2...)。
    # encoding="utf-8-sig": 這是為了讓 Excel 能正確讀取 UTF-8 中文 (加上 BOM)。
    df.to_csv(path, index=False, encoding="utf-8-sig")
    # 紀錄 Log，告知已儲存。
    logging.info(f"已儲存檔案到{pathdir}/")

# [Function 說明]
# 功能：將回測結果或 Log 紀錄存檔。
# 原理：讀取設定檔決定檔名，區分回測結果 (backtests) 與一般紀錄 (logs)。
def result_to_csv(df, is_backtest):
    # 載入設定檔以獲取檔名所需資訊 (如目前的 symbol 和 strategy)。
    config = load_config()
    # 判斷是否為回測結果。
    if is_backtest:
        # 存到 result/backtests 資料夾。
        pathdir = "result/backtests"
    else:
        # 否則存到 result/logs 資料夾。
        pathdir = "result/logs"
    # 從 config 讀取 symbol 和 strategy，從 df 讀取時間範圍來組合成檔名。
    filename = f"{config['基本設定']['symbol']}_{config['基本設定']['strategy']}_{df.loc[0,'時間']} to {df.iloc[-1]['時間']}"
    # 同樣處理檔名中的冒號，避免 Windows 錯誤。
    filename = re.sub(":", "-", filename)
    # 呼叫底層存檔函式。
    to_csv(df, pathdir, filename)

# [Function 說明]
# 功能：根據名稱動態載入策略物件。
# 原理：從 STRATEGY_REGISTRY (註冊表) 中查找字串對應的 Class，並實例化 (Instantiate) 它。
def load_strategy(strategy_name:str):
    #載入策略
    # 印出目前的策略註冊表內容，方便除錯。
    logging.info(STRATEGY_REGISTRY)
    # 檢查請求的策略名稱是否在註冊表中。
    if strategy_name not in STRATEGY_REGISTRY:
        # 如果找不到，拋出 ValueError 錯誤。
        raise ValueError("找不到策略{strategy_name}")
    # 從字典中取出對應的策略類別 (Class)。
    strategy = STRATEGY_REGISTRY[strategy_name]
    # 將類別實例化 (加上括號等於建立物件)，變數 s 現在是一個具體的策略物件。
    s = strategy()
    # 紀錄載入成功的訊息。
    logging.info(f"載入策略{s.name}成功")
    # 回傳策略物件。
    return s

# [Function 說明]
# 功能：讀取已經被策略處理過並存檔的資料。
# 原理：直接從 data/processed 資料夾讀取 CSV。
def get_processed_data(filename:str):
        # 設定讀取路徑。
        pathdir = "data/processed/"
        # 組合完整路徑。
        path = pathdir + filename
        # 檢查檔案是否存在。
        if os.path.lexists(path):
            # 如果存在，紀錄 Log。
            logging.info("正在調用策略訊號")
            # 讀取 CSV 並回傳 DataFrame。
            return pd.read_csv(path)
        else:
            # 如果不存在，拋出錯誤，提示使用者先去跑策略生成資料。
            raise ValueError(f"{filename}不存在，請先使用策略取得訊號資料")