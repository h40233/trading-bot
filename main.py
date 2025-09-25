from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (DerivativesTradingUsdsFutures,ConfigurationRestAPI,DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL)
import logging
from util import *

#logging設定
logging.basicConfig(level=logging.INFO)

#抓config
config = load_config()

#載入策略
strategy = load_strategy(config["strategy"])

#設定datetime
start_time = to_timestamp(config["start_time"])
end_time = to_timestamp(config["end_time"])

#建立client
API_KEY = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
url = config["testnet_url"] if config["testnet"] else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL

config_api=ConfigurationRestAPI(
            api_key = API_KEY,
            api_secret = API_SECRET,
            base_path = url
            )
client = DerivativesTradingUsdsFutures(config_rest_api=config_api).rest_api

#抓K線
df, needsave = get_kline_data(client, config["symbol"], config["timeframe"], start_time, end_time, config["use_mark_price_kline"],config["fetch_limit"],config["sleep_time"])
if needsave:
    data_to_csv(df,is_raw = True)
out = strategy.generate_signal(df)
data_to_csv(out, is_raw = False)
