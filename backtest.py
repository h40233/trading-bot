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
        
        pnl, open_log=self.position.open(close, size*config["leverage"], tp, sl)
        self.stats.trade_log(pnl, open_log)

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
        
        pnl, open_log=self.position.open(close, size*config["leverage"], tp, sl)
        self.stats.trade_log(pnl, open_log)

    def show(self):
        pass

    def run(self):
        for i in tqdm(range(len(self.df))):
            if self.stats.cash <= 0:
                logging.info("資金不足，無法繼續交易")
                break
            row = self.df.iloc[i]
            if self.position.size != 0:
                pnl, close_log = self.position.trigger_SL(row["close"])
                self.stats.trade_log(pnl, close_log)
                pnl, tp_log = self.position.trigger_TP(row["close"])
                self.stats.trade_log(pnl, tp_log)
            if row["signal"] == 1:
                self.buy(row["close"])
                continue
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
        direction = 1 if size > 0 else -1 if size < 0 else 0
        if direction == 0:
            raise ValueError("size不能為0")
        
        if size>0:
            if not (sl or -np.inf) < price < (tp or np.inf):
                raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
        else:
            if not (tp or -np.inf) < price < (sl or np.inf):
                raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
        
        if self.size != 0:
            if self.size * size < 0 :
                # if abs(size) > abs(self.size):
                    #允許反手就反手，否則甚麼都不做
                    if config["reverse"]:
                        pnl, log = self.reverse(price, size, tp, sl)
                        return pnl, log
                    else:
                        return None, None
                # else:
                #     #如果開反方向的小倉位->平一部分倉位
                #     pnl, log = self.close(price, -size)
                #     return pnl, log
            else:
                #如果能同方向加倉就加，否則甚麼都不做進入下一根K棒
                if not config["pyramiding"]:
                    return None, None
        self.avg_price = (self.avg_price * abs(self.size) + price * abs(size)) / (abs(self.size) + abs(size))
        self.size += size
        self.tp = tp
        self.sl = sl
        columns = ["交易序號","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log = [self.index, "開倉", direction, price, size, self.avg_price, self.size]
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
        columns = ["交易序號", "狀態","出場價","出場量","實現損益", "剩餘倉位"]
        log = [self.index, "平倉", price,size, pnl, self.size]
        log = pd.DataFrame([log], columns=columns)
        self.index += 1
        return pnl, log
    
    def close_all(self,
                price:float
                )->tuple[float,pd.DataFrame]:
        """全部平倉"""
        self.sl = None
        self.tp = None
        return self.close(price, self.size)

    def reverse(self,
                price:float,
                size:float,
                tp:float,
                sl:float
                ):
        """反手，先全部平倉再開新倉"""
        pnl_close,log_close = self.close_all(price)
        pnl_open,log_open = self.open(price, size+self.size, tp, sl)
        log = pd.concat([log_close, log_open], ignore_index=True)
        return pnl_close+pnl_open, log
    
    def trigger_SL(self,
                   close
                   ):
            if self.sl is not None:
                if ((close <= self.sl) and (self.size > 0)) or ((close >= self.sl) and (self.size < 0)):
                    return self.close_all(self.sl)
            return None, None
    
    def trigger_TP(self,
                   close
                   )->bool:
        if self.tp is not None:
            if ((close >= self.tp) and (self.size > 0)) or ((close <= self.tp) and (self.size < 0)):
                return self.close_all(self.tp)
        return None, None

class stats:
    """只負責記錄資料的動作"""
    def __init__(self,
                cash:float = 10000
                ):
        self.count = 0
        self.count_long = 0
        self.count_long_win = 0
        self.count_short = 0
        self.count_short_win = 0
        self.log=pd.DataFrame()
        self.cash = cash
        self.pnl = 0
        self.max_drawdown = 0

    def trade_log(self,
                pnl:float,
                log:pd.DataFrame
                ):
        if log is not None:
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
                    if pnl > 0:
                        self.count_long_win += 1
                else:
                    self.count_short += 1
                    if pnl > 0:
                        self.count_short_win += 1

    def sharpe():
        pass

    def long_winrate(self):
        if self.count_long == 0:
            return 0
        return self.count_long_win / self.count_long * 100

    def short_winrate(self):
        if self.count_short == 0:
            return 0
        return self.count_short_win / self.count_short * 100

    def winrate(self):
        if self.count == 0:
            return 0
        return (self.count_long_win + self.count_short_win) / self.count * 100

if __name__ == "__main__":
    bt = backtest(df)
    bt.run()