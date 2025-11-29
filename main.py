from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (DerivativesTradingUsdsFutures,ConfigurationRestAPI,DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL)
import logging
from util import *

def main():
    """主執行函式"""
    #logging設定
    logging.basicConfig(level=logging.INFO)

    #抓config
    config = load_config()

    #載入策略
    strategy = load_strategy(config["基本設定"]["strategy"])

    #設定datetime
    start_time = to_timestamp(config["回測設定"]["start_time"])
    end_time = to_timestamp(config["回測設定"]["end_time"])

    #建立client
    API_KEY = os.getenv("API_KEY","")
    API_SECRET = os.getenv("API_SECRET","")
    url = config["基本設定"]["testnet_url"] if config["基本設定"]["testnet"] else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL

    config_api=ConfigurationRestAPI(
                api_key = API_KEY,
                api_secret = API_SECRET,
                base_path = url
                )
    client = DerivativesTradingUsdsFutures(config_rest_api=config_api).rest_api

    #抓K線
    df, needsave = get_kline_data(client, config["基本設定"]["symbol"], config["基本設定"]["timeframe"], start_time, end_time, config["基本設定"]["use_mark_price_kline"],config["基本設定"]["fetch_limit"],config["基本設定"]["sleep_time"])
    if needsave:
        data_to_csv(df,is_raw = True)
    out = strategy.generate_signal(df)
    data_to_csv(out, is_raw = False)

if __name__ == "__main__":
    main()
