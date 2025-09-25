from util import *
import logging
import numpy as np
from tqdm import tqdm

#logging設定
logging.basicConfig(level=logging.INFO)

#載入config
config = load_config()
pathdir = "result/backtests"
filename = f"{config["symbol"]}_{config["strategy"]}_{config["start_time"]} to {config["end_time"]}.csv"
filename = re.sub(":","-",filename)
df = get_processed_data(filename)

class backtest:
    """只負責呼叫其他方法"""
    def __init__(self, df):
        self.df = df
        self.position = position()
        self.stats = stats(config["initial_cash"])

    def buy(self, close):
        if config["tp_of_percent"]:
            tp = close*(1+config["tp_value"]/100)
        else:
            tp = close+config["tp_value"]
        
        if config["sl_of_percent"]:
            sl = close*(1-config["sl_value"]/100)
        else:
            sl = close-config["sl_value"]
        
        if config["order_mode"] == "percent":
            size = self.stats.cash * config["order_value"] / 100 / close
        elif config["order_mode"] == "price":
            size = config["order_value"] / close
        elif config["order_mode"] == "fixed":
            size = config["order_value"]
        else:
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")
        
        pnl, open=self.position.open(close, size, tp, sl)
        self.stats.trade_log(pnl, open)

    def sell(self, close):
        if config["tp_of_percent"]:
            tp = close*(1-config["tp_value"]/100)
        else:
            tp = close-config["tp_value"]
        
        if config["sl_of_percent"]:
            sl = close*(1+config["sl_value"]/100)
        else:
            sl = close+config["sl_value"]
        
        if config["order_mode"] == "percent":
            size = - self.stats.cash * config["order_value"] / 100 / close
        elif config["order_mode"] == "price":
            size = - config["order_value"] / close
        elif config["order_mode"] == "fixed":
            size = - config["order_value"]
        else:
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")
        
        pnl, open=self.position.open(close, size, tp, sl)
        self.stats.trade_log(pnl, open)
    
    def show(self):
        pass
    def run(self):
        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[i]
            if row["signal"] == 1:
                self.buy(row["close"])
            elif row["signal"] == -1:
                self.sell(row["close"])
        if self.position.size != 0:
            pnl, close = self.position.close_all(self.df.iloc[-1]["close"])
            self.stats.trade_log(pnl, close)
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl}, 最大回撤: {self.stats.max_drawdown}, 最終資金: {self.stats.cash}")
        to_csv(self.stats.log, pathdir, filename)

class position:
    """只負責倉位部分的動作"""
    def __init__(self):
        self.avg_price = 0
        self.size = 0
        self.index = 1

    def open(self,
            price:float,
            size:float,
            tp:float,
            sl:float
            )->tuple[float,pd.DataFrame]:
        """開倉，回傳pnl和log"""
        direction =1 if size>0 else -1
        if size>0:
            if not (sl or -np.inf) < price < (tp or np.inf):
                raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
            if self.size < 0:
                return self.reverse(price, size, tp, sl)
        else:
            if not (tp or -np.inf) < price < (sl or np.inf):
                raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
            if self.size > 0:
                return self.reverse(price, size, tp, sl)
        self.avg_price = (self.avg_price * abs(self.size) + price * abs(size)) / (abs(self.size) + abs(size))
        self.size += size
        self.tp = tp
        self.sl = sl
        columns = ["交易序號","多/空","進場價","進場量","當前均價","當前持倉量","實現損益"]
        log = [self.index, direction, price, size, self.avg_price, self.size, 0]
        log = pd.DataFrame([log], columns=columns)
        self.index += 1
        return 0,log

    def close(self,
            price:float,
            size:float,
            )->tuple[float,pd.DataFrame]:
        """平倉，回傳pnl和log"""
        if self.size == 0:
            raise Exception("當前無持倉，無法平倉")
        pnl = (price - self.avg_price)*size
        self.size -= size
        if self.size == 0:
            self.avg_price = 0
        columns = ["交易序號","出場價","實現損益"]
        log = [self.index, price, pnl]
        log = pd.DataFrame([log], columns=columns)
        self.index += 1
        return pnl, log
    
    def close_all(self,
                price:float
                )->tuple[float,pd.DataFrame]:
        """全部平倉，回傳pnl"""
        return self.close(price, self.size)

    def reverse(self,
                price:float,
                size:float,
                tp:float,
                sl:float
                ):
        """反手，先全部平倉再開新倉，回傳log和pnl"""
        pnl_close,log_close = self.close_all(price)
        pnl_open,log_open = self.open(price, size+self.size, tp, sl)
        log = pd.concat([log_close, log_open], ignore_index=True)
        return pnl_close+pnl_open, log
    
    def trigger_SL(self)->bool:
        pass

    def trigger_TP(self)->bool:
        pass

class stats:
    """只負責記錄資料的動作"""
    def __init__(self,
                cash:float = 10000
                ):
        self.count = 0
        self.count_long = 0
        self.count_short = 0
        self.log=pd.DataFrame()
        self.cash = cash
        self.pnl = 0
        self.max_drawdown = 0

    def trade_log(self,
                pnl:float,
                log:pd.DataFrame
                ):
        self.count += 1
        self.pnl += pnl
        self.cash += pnl
        if self.pnl < self.max_drawdown:
            self.max_drawdown = self.pnl
        self.log = pd.concat([self.log, log], ignore_index=True)
        if "多/空" in log.columns:
            side = log["多/空"].iloc[-1]
            if side == 1:
                self.count_long += 1
            else:
                self.count_short += 1

    def shape():
        pass

    def long_winrate(self):
        pass

    def short_winrate(self):
        pass

    def winrate(self):
        pass

if __name__ == "__main__":
    bt = backtest(df)
    bt.run()