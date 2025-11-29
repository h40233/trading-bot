from util import *
import logging
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import numpy as np
from tqdm import tqdm

#logging設定
logging.basicConfig(level=logging.INFO)

class backtest:
    # 設定 Decimal 的精度
    getcontext().prec = 28

    """只負責呼叫其他方法"""
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.position = position(self.config)
        self.stats = stats(self.config)
        self.max_hold = self.config["基本設定"].get("max_hold", None) # 從設定檔讀取最長持倉K棒數

    def _create_order(self, close: Decimal, direction: int, timestamp, i: int):
        """內部方法，根據方向創建訂單"""
        if direction not in [1, -1]:
            raise ValueError("direction 必須是 1 (多) 或 -1 (空)")

        if self.config["止盈止損設定"]["tp_of_percent"]:
            tp = close * (Decimal('1') + Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["tp_value"])) / Decimal('100'))
        else:
            tp = close + Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["tp_value"]))

        if self.config["止盈止損設定"]["sl_of_percent"]:
            sl = close * (Decimal('1') - Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["sl_value"])) / Decimal('100'))
        else:
            sl = close - Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["sl_value"]))

        # 對計算出的止盈止損價格進行四捨五入，避免浮點數精度問題
        tp = tp.quantize(Decimal('1e-8'))
        sl = sl.quantize(Decimal('1e-8'))

        if self.config["下單設定"]["order_mode"] == "percent":
            base_size = self.stats.cash * Decimal(str(self.config["下單設定"]["order_value"])) / Decimal('100') / close
        elif self.config["下單設定"]["order_mode"] == "price":
            base_size = Decimal(str(self.config["下單設定"]["order_value"])) / close
        elif self.config["下單設定"]["order_mode"] == "fixed":
            base_size = Decimal(str(self.config["下單設定"]["order_value"]))
        else:
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")

        size = Decimal(str(direction)) * base_size * Decimal(str(self.config["下單設定"]["leverage"]))

        # --- 資金不足判斷 ---
        # 計算開倉所需保證金 (未計槓桿的價值) + 開倉手續費
        margin = close * base_size
        # 預估手續費，用於資金檢查
        estimated_fee = margin * Decimal(str(self.config["下單設定"]["leverage"])) * self.position.fee_rate
        if self.stats.cash < (margin + estimated_fee):
            logging.warning(f"[INSUFFICIENT_FUNDS] 時間: {timestamp}, 資金 {self.stats.cash:.2f} 不足，無法開倉 (需要 {margin+estimated_fee:.2f})")
            return # 直接返回，不執行後續開倉

        # 呼叫 position.open，它現在可能回傳多個日誌事件 (例如反手時)
        logs_to_process = self.position.open(close, size, tp, sl, timestamp, i)

        # 遍歷所有產生的日誌事件並逐一記錄
        if logs_to_process:
            for pnl, log_event in logs_to_process:
                self.stats.trade_log(pnl, log_event)

    def show(self):
        logging.info(f"===== 績效總結 =====")
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl:.2f}, 最終資金: {self.stats.cash:.2f}, 最大回撤: {self.stats.max_drawdown:.2f}")
        logging.info(f"勝率 (總/多/空): {self.stats.winrate():.2f}% / {self.stats.long_winrate():.2f}% / {self.stats.short_winrate():.2f}%")
        logging.info(f"獲利因子: {self.stats.profit_factor():.2f}, 夏普比率: {self.stats.sharpe():.2f}, 索提諾比率: {self.stats.sortino_ratio():.2f}, 卡瑪比率: {self.stats.calmar_ratio():.2f}")

    def plot_results(self):
        """繪製並儲存資金曲線圖"""
        self.stats.plot_equity_curve()

    def run(self):
        for i in tqdm(range(len(self.df))):
            if self.stats.cash <= 0:
                logging.info("資金不足，無法繼續交易")
                break
            row = self.df.iloc[i].copy() # 使用 .copy() 避免 SettingWithCopyWarning
            row['close'] = Decimal(str(row['close'])) # 將價格轉換為 Decimal
            if self.position.size != 0:
                logs_sl = self.position.trigger_SL(row["close"], row["close_time"])
                if logs_sl:
                    for pnl, log_event in logs_sl:
                        self.stats.trade_log(pnl, log_event)

                # --- 強制平倉判斷 ---
                # 如果設定了最長持倉時間，並且當前K棒索引 - 開倉時索引 > max_hold
                if self.max_hold is not None and (i - self.position.entry_index) >= self.max_hold:
                    logs_force_close = self.position.close_all(row["close"], row["close_time"])
                    if logs_force_close:
                        for pnl, log_event in logs_force_close:
                            self.stats.trade_log(pnl, log_event)

                logs_tp = self.position.trigger_TP(row["close"], row["close_time"])
                if logs_tp:
                    for pnl, log_event in logs_tp:
                        self.stats.trade_log(pnl, log_event)
            # 無論之前是否平倉，都檢查當前K棒是否有開倉信號
            if row["signal"] == 1:
                self._create_order(row["close"], 1, row["close_time"], i)
            elif row["signal"] == -1:
                self._create_order(row["close"], -1, row["close_time"], i)
        if self.position.size != 0:
            pnl, close = self.position.close_all(self.df.iloc[-1]["close"], self.df.iloc[-1]["close_time"])
            self.stats.trade_log(pnl, close)
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl}, 最大回撤: {self.stats.max_drawdown}, 最終資金: {self.stats.cash}")
        # 只有在 log 不是空的情況下才儲存結果
        if not self.stats.log.empty:
            result_to_csv(self.stats.log, is_backtest=True)

class position:
    """只負責倉位部分的動作"""
    def __init__(self, config):
        self.config = config
        self.avg_price = Decimal('0')
        self.size = Decimal('0')
        self.entry_index = 0 # 記錄開倉時的K棒索引
        self.fee_rate = Decimal(str(self.config["回測設定"]["fee_rate"]))
        self.slippage = Decimal(str(self.config["回測設定"]["slippage"]))

    def open(self,
            price: Decimal,
            size: Decimal,
            tp: Decimal,
            sl: Decimal,
            timestamp,
            entry_index: int
            ) -> list[tuple[Decimal, pd.DataFrame]]:
        """開倉，回傳pnl和log"""
        direction = Decimal('1') if size > Decimal('0') else Decimal('-1') if size < Decimal('0') else Decimal('0')
        if direction == Decimal('0'):
            raise ValueError("size不能為0")
        
        if size > Decimal('0'):
            if not (sl or Decimal('-Infinity')) < price < (tp or Decimal('Infinity')):
                raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
            price = price * (Decimal('1') + self.slippage)
        else:
            if not (tp or Decimal('-Infinity')) < price < (sl or Decimal('Infinity')):
                raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
            price = price * (Decimal('1') - self.slippage)
        
        if self.size != Decimal('0'):
            if self.size * size < Decimal('0'):
                if self.config["下單設定"]["reverse"]:
                    return self.reverse(price, size, tp, sl, timestamp, entry_index)
                return [] # 如果不允許反手，則返回空列表
                # else:
                #     #如果開反方向的小倉位->平一部分倉位
                #     pnl, log = self.close(price, -size)
                #     return pnl, log
            else:
                #如果能同方向加倉就加，否則甚麼都不做進入下一根K棒
                if not self.config["下單設定"]["pyramiding"]:
                    return [] # 如果不允許加倉，則返回空列表
        self.avg_price = (self.avg_price * self.size.copy_abs() + price * size.copy_abs()) / (self.size.copy_abs() + size.copy_abs())
        self.size += size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index # 設定開倉索引

        columns = ["時間","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log = pd.DataFrame([[timestamp, "開倉", direction, price, size, self.avg_price, self.size]], columns=columns)
        return [(Decimal('0'), log)] # 作為列表返回

    def close(self,
            price:float,
            size:float, # 注意: 這裡的size是帶有方向的平倉量
            timestamp
            ) -> list[tuple[Decimal, pd.DataFrame]]:
        """平倉，回傳pnl和log"""
        if self.size == Decimal('0'):
            raise Exception("當前無持倉，無法平倉")
        if size > Decimal('0'):
            price = price * (Decimal('1') - self.slippage)
        else:
            price = price * (Decimal('1') + self.slippage)

        # 修正後的損益計算:
        # PnL = 毛利 - 平倉手續費。開倉手續費在開倉時就已記錄。
        gross_pnl = (price - self.avg_price) * (-size)
        closing_fee = size.copy_abs() * price * self.fee_rate
        pnl = gross_pnl - closing_fee
        self.size += size
        if self.size == Decimal('0'):
            self.avg_price = Decimal('0')
        columns = ["時間", "狀態","出場價","出場量","實現損益", "剩餘倉位"]
        log = pd.DataFrame([[timestamp, "平倉", price, -size, pnl, self.size]], columns=columns)
        return [(pnl, log)]
    
    def close_all(self,
                price:float,
                timestamp
                ) -> list[tuple[float, pd.DataFrame]]:
        """全部平倉"""
        if self.size == 0:
            return []
        self.sl = None
        self.tp = None
        return self.close(price, -self.size, timestamp)

    def reverse(self,
                price: Decimal,
                size: Decimal,
                tp: Decimal,
                sl: Decimal,
                timestamp,
                entry_index:int
                ) -> list[tuple[Decimal, pd.DataFrame]]:
        """反手，先全部平倉再開新倉，返回一個包含兩次操作日誌的列表"""
        # 1. 平倉
        close_results = self.close_all(price, timestamp)
        if not close_results:
            return [] # 如果平倉失敗或沒有倉位，直接返回
        pnl_close, log_close = close_results[0]
        # 2. 開新倉
        # 注意：這裡直接遞歸呼叫 open，因為此時 self.size 已經為 0
        # open_results = self.open(price, size, tp, sl, timestamp, entry_index)
        # return [(pnl_close, log_close)] + open_results
        direction = Decimal('1') if size > Decimal('0') else Decimal('-1')
        self.avg_price = price
        self.size = size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index
        columns = ["時間","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log_open = pd.DataFrame([[timestamp, "開倉", direction, price, size, self.avg_price, self.size]], columns=columns)
        return [(pnl_close, log_close), (Decimal('0'), log_open)]

    def trigger_SL(self,
                   close: Decimal, timestamp
                   ):
        if self.sl is not None:
            if ((close <= self.sl) and (self.size > Decimal('0'))) or ((close >= self.sl) and (self.size < Decimal('0'))):
                return self.close_all(self.sl, timestamp)
        return []
    
    def trigger_TP(self,
                   close: Decimal, timestamp
                   ):
        if self.tp is not None:
            if ((close >= self.tp) and (self.size > Decimal('0'))) or ((close <= self.tp) and (self.size < Decimal('0'))):
                return self.close_all(self.tp, timestamp)
        return []

class stats:
    """只負責記錄資料的動作"""
    def __init__(self, config):
        self.config = config
        self.count = Decimal('0')
        self.count_long = Decimal('0')
        self.count_long_win = Decimal('0')
        self.count_short = Decimal('0')
        self.count_short_win = Decimal('0')
        self.log = pd.DataFrame()
        self.cash = Decimal(str(self.config["回測設定"]["initial_cash"]))
        self.pnl = Decimal('0')
        self.max_drawdown = Decimal('0')

    def trade_log(self,
                pnl: Decimal,
                log:pd.DataFrame
                ):
        if log is not None:
            self.log = pd.concat([self.log, log], ignore_index=True)

            # 如果是開倉日誌，直接扣除開倉手續費
            if "狀態" in log.columns and log["狀態"].iloc[0] == "開倉":
                opening_fee = log["進場量"].iloc[0].copy_abs() * log["進場價"].iloc[0] * Decimal(str(self.config["回測設定"]["fee_rate"]))
                self.cash -= opening_fee
                # 開倉本身不計入總pnl，手續費直接從現金扣除

            # 如果是平倉日誌，才更新損益和交易統計
            elif "狀態" in log.columns and log["狀態"].iloc[0] == "平倉":
                if pnl is None or pnl == Decimal('0'):
                    return # 如果平倉事件沒有產生損益，則不進行統計

                self.pnl += pnl
                self.cash += pnl
                # 最大回撤應該基於資金曲線，這裡用累計pnl做簡化計算
                if self.pnl < self.max_drawdown:
                    self.max_drawdown = self.pnl

                closed_size = log["出場量"].iloc[0]
                self.count += Decimal('1')
                if closed_size > Decimal('0'): # 平多倉
                    self.count_long += Decimal('1')
                    if pnl > Decimal('0'): self.count_long_win += Decimal('1')
                elif closed_size < Decimal('0'): # 平空倉
                    self.count_short += Decimal('1')
                    if pnl > Decimal('0'): self.count_short_win += Decimal('1')

    def sharpe(self):
        # 複製一份log以避免修改原始數據
        log_df = self.log.copy()

        # 確保 '實現損益' 欄位存在且有非NaN值
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return Decimal('0.0')

        # 將時間字串轉換為datetime物件，並設為索引
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        log_df.set_index('時間', inplace=True)

        # 按天對損益進行分組加總，得到每日報酬序列
        daily_returns = log_df['實現損益'].astype(float).resample('D').sum()

        # 如果回測期間小於2天，無法計算標準差，返回0
        if len(daily_returns) < 2:
            return Decimal('0.0')

        # 計算年化夏普比率
        rf = Decimal('0.01')
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # 如果標準差為0 (例如每天都沒賺沒賠)，返回0避免除以零錯誤
        if std_daily_return == 0:
            return Decimal('0.0')

        # (每日平均報酬 - 每日無風險利率) / 每日報酬標準差 * 年化因子(sqrt(252))
        daily_rf = rf / 252
        sharpe_ratio = (mean_daily_return - daily_rf) / std_daily_return * np.sqrt(252)
        return sharpe_ratio

    def get_equity_curve(self):
        """計算並返回資金曲線的時間序列數據"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return None

        # 計算每個時間點的累計損益
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        pnl_events = log_df.dropna(subset=['實現損益'])
        pnl_events = pnl_events.sort_values(by='時間')
        pnl_events['累計損益'] = pnl_events['實現損益'].cumsum()
        pnl_events['資金曲線'] = self.cash - self.pnl + pnl_events['累計損益']

        return pnl_events[['時間', '資金曲線']]

    def plot_equity_curve(self):
        """繪製資金曲線圖"""
        equity_df = self.get_equity_curve()
        if equity_df is None:
            logging.warning("沒有足夠的交易數據來繪製資金曲線圖。")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['時間'], equity_df['資金曲線'], label='Equity Curve')
        plt.title(f"Equity Curve - {self.config['基本設定']['symbol']} - {self.config['基本設定']['strategy']}")
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        plt.show()

    def long_winrate(self):
        if self.count_long == Decimal('0'):
            return Decimal('0')
        return (self.count_long_win / self.count_long) * Decimal('100')

    def short_winrate(self):
        if self.count_short == Decimal('0'):
            return Decimal('0')
        return (self.count_short_win / self.count_short) * Decimal('100')

    def winrate(self):
        if self.count == Decimal('0'):
            return Decimal('0')
        return ((self.count_long_win + self.count_short_win) / self.count) * Decimal('100')

    def profit_factor(self):
        """計算獲利因子 (總盈利 / 總虧損)"""
        if "實現損益" not in self.log.columns:
            return Decimal('0.0')

        returns = self.log["實現損益"].dropna()
        total_profit = sum(r for r in returns if r > Decimal('0'))
        total_loss = sum(r for r in returns if r < Decimal('0')).copy_abs()

        if total_loss == Decimal('0'):
            return Decimal('Infinity') # 如果沒有虧損，獲利因子為無限大

        return total_profit / total_loss

    def sortino_ratio(self):
        """計算年化索提諾比率"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return Decimal('0.0')

        log_df['時間'] = pd.to_datetime(log_df['時間'])
        log_df.set_index('時間', inplace=True)
        daily_returns = log_df['實現損益'].astype(float).resample('D').sum()

        if len(daily_returns) < 2:
            return Decimal('0.0')

        rf = Decimal('0.01')
        mean_daily_return = daily_returns.mean()
        
        # 計算下行標準差 (只考慮虧損日的波動)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return Decimal('0.0')

        daily_rf = rf / 252
        sortino = (mean_daily_return - daily_rf) / downside_std * np.sqrt(252)
        return sortino

    def calmar_ratio(self):
        """計算卡瑪比率 (年化報酬率 / 最大回撤)"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return 0.0

        # 計算年化報酬率
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        start_date = log_df['時間'].min()
        end_date = log_df['時間'].max()
        num_days = (end_date - start_date).days
        
        if num_days == 0:
            return 0.0

        # 總報酬率
        total_return_rate = self.pnl / self.config["回測設定"]["initial_cash"]
        # 年化報酬率
        annualized_return = (1 + total_return_rate) ** (365.0 / num_days) - 1

        # 取得最大回撤的絕對值
        # 注意：目前的最大回撤是基於累計PnL，更精確的應基於資金曲線，但此處使用現有值
        max_dd_value = abs(self.max_drawdown)
        if max_dd_value == 0:
            return float('inf') # 如果沒有回撤，比率為無限大

        max_dd_percent = max_dd_value / self.config["回測設定"]["initial_cash"] # 將回撤轉換為百分比

        return annualized_return / max_dd_percent

if __name__ == "__main__":
    # 將所有執行邏輯移至此處
    #載入config
    config = load_config()
    pathdir = "result/backtests"
    filename = f"{config['基本設定']['symbol']}_{config['基本設定']['strategy']}_{config['回測設定']['start_time']} to {config['回測設定']['end_time']}.csv"
    filename = re.sub(":","-",filename)
    df = get_processed_data(filename)

    bt = backtest(df, config)
    bt.run()
    bt.show()
    bt.plot_results()