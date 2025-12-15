# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案是「回測引擎 (Backtesting Engine)」。
# 它的功能是模擬真實的市場環境，逐行讀取 K 線資料，執行策略信號，並計算損益。
# 它是完全獨立的，不連接交易所，只在本地運算。
# -----------------------------------------------------------------------------------------

# [Import 說明]
from util import *
import logging
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import numpy as np
from tqdm import tqdm
import pandas as pd

# 設定日誌輸出的等級為 INFO。
logging.basicConfig(level=logging.INFO)

# [Class 說明]
# 職責：回測的主控制器 (Controller)。
class backtest:
    # 設定 Decimal 的運算精度為 28 位。
    getcontext().prec = 28

    """只負責呼叫其他方法"""
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.position = position(self.config)
        self.stats = stats(self.config)
        self.max_hold = self.config["基本設定"].get("max_hold", None)

    def _create_order(self, close: Decimal, direction: int, timestamp, i: int):
        """內部方法，根據方向創建訂單"""
        if direction not in [1, -1]:
            raise ValueError("direction 必須是 1 (多) 或 -1 (空)")

        # --- 計算止盈 (TP) ---
        tp_value = Decimal(str(self.config["止盈止損設定"]["tp_value"]))
        if self.config["止盈止損設定"]["tp_of_percent"]:
            tp = close * (Decimal('1') + Decimal(str(direction)) * tp_value / Decimal('100'))
        else:
            tp = close + Decimal(str(direction)) * tp_value

        # --- 計算止損 (SL) ---
        sl_value = Decimal(str(self.config["止盈止損設定"]["sl_value"]))
        if self.config["止盈止損設定"]["sl_of_percent"]:
            sl = close * (Decimal('1') - Decimal(str(direction)) * sl_value / Decimal('100'))
        else:
            sl = close - Decimal(str(direction)) * sl_value

        tp = tp.quantize(Decimal('1e-8'))
        sl = sl.quantize(Decimal('1e-8'))

        # --- 計算下單數量 (Size) ---
        order_val = Decimal(str(self.config["下單設定"]["order_value"]))
        if self.config["下單設定"]["order_mode"] == "percent":
            base_size = self.stats.cash * order_val / Decimal('100') / close
        elif self.config["下單設定"]["order_mode"] == "price":
            base_size = order_val / close
        elif self.config["下單設定"]["order_mode"] == "fixed":
            base_size = order_val
        else:
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")

        leverage = Decimal(str(self.config["下單設定"]["leverage"]))
        size = Decimal(str(direction)) * base_size * leverage

        # --- 資金不足判斷 ---
        margin = close * base_size
        estimated_fee = margin * leverage * self.position.fee_rate
        
        if self.stats.cash < (margin + estimated_fee):
            logging.warning(f"[INSUFFICIENT_FUNDS] 時間: {timestamp}, 資金 {self.stats.cash:.2f} 不足，無法開倉 (需要 {margin+estimated_fee:.2f})")
            return 

        logs_to_process = self.position.open(close, size, tp, sl, timestamp, i)

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

    def run(self, progress_callback=None):
        # 取得資料總長度
        total_len = len(self.df)
        
        # 使用 tqdm 建立進度條 (這是給終端機看的)
        for i in tqdm(range(total_len)):
            
            # --- 更新 Streamlit 進度條 ---
            if progress_callback:
                if i % (total_len // 100 + 1) == 0:
                    progress = i / total_len
                    progress_callback(progress)
            # -------------------------------------

            # 檢查破產保護
            if self.stats.cash <= 0:
                logging.info("資金不足，無法繼續交易")
                break
            
            row = self.df.iloc[i].copy() 
            row['close'] = Decimal(str(row['close'])) 
            
            if self.position.size != 0:
                # 1. 檢查是否觸發止損 (SL)。
                logs_sl = self.position.trigger_SL(row["close"], row["close_time"])
                if logs_sl:
                    for pnl, log_event in logs_sl:
                        self.stats.trade_log(pnl, log_event)

                # --- 強制平倉判斷 (Max Hold) ---
                if self.max_hold is not None and (i - self.position.entry_index) >= self.max_hold:
                    logs_force_close = self.position.close_all(row["close"], row["close_time"])
                    if logs_force_close:
                        for pnl, log_event in logs_force_close:
                            self.stats.trade_log(pnl, log_event)

                # 2. 檢查是否觸發止盈 (TP)。
                logs_tp = self.position.trigger_TP(row["close"], row["close_time"])
                if logs_tp:
                    for pnl, log_event in logs_tp:
                        self.stats.trade_log(pnl, log_event)
            
            # 檢查進場信號。
            if row["signal"] == 1:
                self._create_order(row["close"], 1, row["close_time"], i)
            elif row["signal"] == -1:
                self._create_order(row["close"], -1, row["close_time"], i)
        
        # 迴圈結束後平倉
        if self.position.size != 0:
            last_price = Decimal(str(self.df.iloc[-1]["close"]))
            close_results = self.position.close_all(last_price, self.df.iloc[-1]["close_time"])
            if close_results:
                pnl, log = close_results[0]
                self.stats.trade_log(pnl, log)
        
        if progress_callback:
            progress_callback(1.0)
        
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl}, 最大回撤: {self.stats.max_drawdown}, 最終資金: {self.stats.cash}")
        
        if not self.stats.log.empty:
            result_to_csv(self.stats.log, is_backtest=True)

# [Class 說明]
# 職責：管理單一倉位 (Position) 的狀態。
class position:
    """只負責倉位部分的動作"""
    def __init__(self, config):
        self.config = config
        self.avg_price = Decimal('0')
        self.size = Decimal('0')
        self.entry_index = 0
        self.fee_rate = Decimal(str(self.config["回測設定"]["fee_rate"]))
        self.slippage = Decimal(str(self.config["回測設定"]["slippage"]))

    def open(self, price: Decimal, size: Decimal, tp: Decimal, sl: Decimal, timestamp, entry_index: int) -> list[tuple[Decimal, pd.DataFrame]]:
        """開倉，回傳pnl和log"""
        direction = Decimal('1') if size > Decimal('0') else Decimal('-1') if size < Decimal('0') else Decimal('0')
        if direction == Decimal('0'):
            raise ValueError("size不能為0")
        
        if size > Decimal('0'): # 做多
            if not (sl or Decimal('-Infinity')) < price < (tp or Decimal('Infinity')):
                raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
            price = price * (Decimal('1') + self.slippage)
        else: # 做空
            if not (tp or Decimal('-Infinity')) < price < (sl or Decimal('Infinity')):
                raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
            price = price * (Decimal('1') - self.slippage)
        
        if self.size != Decimal('0'):
            if self.size * size < Decimal('0'):
                if self.config["下單設定"]["reverse"]:
                    return self.reverse(price, size, tp, sl, timestamp, entry_index)
                return [] 
            else:
                if not self.config["下單設定"]["pyramiding"]:
                    return [] 
        
        self.avg_price = (self.avg_price * self.size.copy_abs() + price * size.copy_abs()) / (self.size.copy_abs() + size.copy_abs())
        self.size += size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index 

        columns = ["時間","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log = pd.DataFrame([[timestamp, "開倉", direction, price, size, self.avg_price, self.size]], columns=columns)
        return [(Decimal('0'), log)] 

    def close(self, price:float, size:float, timestamp) -> list[tuple[Decimal, pd.DataFrame]]:
        """平倉，回傳pnl和log"""
        if self.size == Decimal('0'):
            raise Exception("當前無持倉，無法平倉")
        
        if size > Decimal('0'): 
            price = price * (Decimal('1') + self.slippage) 
        else: 
            price = price * (Decimal('1') - self.slippage) 

        gross_pnl = (price - self.avg_price) * (-size)
        closing_fee = size.copy_abs() * price * self.fee_rate
        pnl = gross_pnl - closing_fee
        
        self.size += size
        if self.size == Decimal('0'):
            self.avg_price = Decimal('0')
            
        columns = ["時間", "狀態","出場價","出場量","實現損益", "剩餘倉位"]
        log = pd.DataFrame([[timestamp, "平倉", price, -size, pnl, self.size]], columns=columns)
        return [(pnl, log)]
    
    def close_all(self, price:float, timestamp) -> list[tuple[float, pd.DataFrame]]:
        """全部平倉"""
        if self.size == 0:
            return []
        self.sl = None
        self.tp = None
        return self.close(price, -self.size, timestamp)

    def reverse(self, price: Decimal, size: Decimal, tp: Decimal, sl: Decimal, timestamp, entry_index:int) -> list[tuple[Decimal, pd.DataFrame]]:
        """反手"""
        close_results = self.close_all(price, timestamp)
        if not close_results:
            return []
        
        pnl_close, log_close = close_results[0]
        
        direction = Decimal('1') if size > Decimal('0') else Decimal('-1')
        self.avg_price = price
        self.size = size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index
        
        columns = ["時間","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log_open = pd.DataFrame([[timestamp, "開倉", direction, price, size, self.avg_price, self.size]], columns=columns)
        
        return [(pnl_close, log_close), (Decimal('0'), log_open)]

    def trigger_SL(self, close: Decimal, timestamp):
        if self.sl is not None:
            if ((close <= self.sl) and (self.size > Decimal('0'))) or ((close >= self.sl) and (self.size < Decimal('0'))):
                return self.close_all(self.sl, timestamp)
        return []
    
    def trigger_TP(self, close: Decimal, timestamp):
        if self.tp is not None:
            if ((close >= self.tp) and (self.size > Decimal('0'))) or ((close <= self.tp) and (self.size < Decimal('0'))):
                return self.close_all(self.tp, timestamp)
        return []

# [Class 說明]
# 職責：統計與會計模組 (Statistics)。
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

    def trade_log(self, pnl: Decimal, log:pd.DataFrame):
        if log is not None:
            self.log = pd.concat([self.log, log], ignore_index=True)

            if "狀態" in log.columns and log["狀態"].iloc[0] == "開倉":
                opening_fee = log["進場量"].iloc[0].copy_abs() * log["進場價"].iloc[0] * Decimal(str(self.config["回測設定"]["fee_rate"]))
                
                # [修改點] 同時扣除現金與損益
                self.cash -= opening_fee
                self.pnl -= opening_fee  # <--- 新增這行

            elif "狀態" in log.columns and log["狀態"].iloc[0] == "平倉":
                if pnl is None or pnl == Decimal('0'):
                    return

                self.pnl += pnl
                self.cash += pnl
                
                if self.pnl < self.max_drawdown:
                    self.max_drawdown = self.pnl

                closed_size = log["出場量"].iloc[0]
                self.count += Decimal('1')
                
                if closed_size < Decimal('0'): # 賣出平倉 (原本是多單)
                    self.count_long += Decimal('1')
                    if pnl > Decimal('0'): self.count_long_win += Decimal('1')
                elif closed_size > Decimal('0'): # 買入平倉 (原本是空單)
                    self.count_short += Decimal('1')
                    if pnl > Decimal('0'): self.count_short_win += Decimal('1')

    def sharpe(self):
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return 0.0 

        log_df['時間'] = pd.to_datetime(log_df['時間'])
        log_df.set_index('時間', inplace=True)

        daily_returns = log_df['實現損益'].astype(float).resample('D').sum()

        if len(daily_returns) < 2:
            return 0.0

        rf = 0.01 
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        if std_daily_return == 0:
            return 0.0

        daily_rf = rf / 252
        sharpe_ratio = (mean_daily_return - daily_rf) / std_daily_return * np.sqrt(252)
        return sharpe_ratio

    def get_equity_curve(self):
        """計算並返回資金曲線的時間序列數據"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return None

        log_df['時間'] = pd.to_datetime(log_df['時間'])
        pnl_events = log_df.dropna(subset=['實現損益'])
        pnl_events = pnl_events.sort_values(by='時間')
        
        pnl_events['累計損益'] = pnl_events['實現損益'].apply(float).cumsum()
        
        initial_cash_float = float(self.config["回測設定"]["initial_cash"])
        
        # [修改點] 因為 self.pnl 已經扣過開倉手續費了，但這裡是用平倉紀錄累加的，
        # 如果要非常精確的曲線，理論上開倉當下資金也會掉一點點。
        # 但為了繪圖簡單，這裡邏輯維持：初始資金 + 累計已實現損益
        # 注意：這裡畫出來的圖，最終點可能會比 self.cash 稍微多一點點 (因為還沒扣最後一次開倉費? 不對，這裡只有平倉紀錄)
        # 其實最準確的做法是把開倉紀錄也畫進去，但那樣圖會變得很密。
        # 目前這樣畫是可以接受的近似值。
        pnl_events['資金曲線'] = initial_cash_float + pnl_events['累計損益']

        return pnl_events[['時間', '資金曲線']]

    def plot_equity_curve(self):
        """繪製資金曲線圖"""
        equity_df = self.get_equity_curve()
        if equity_df is None:
            logging.warning("沒有足夠的交易數據來繪製資金曲線圖。")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['時間'], equity_df['資金曲線'], label='Equity Curve')
        plt.title(f"Equity Curve - {self.config['基本設定']['symbol']}")
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
        """計算獲利因子"""
        if "實現損益" not in self.log.columns:
            return Decimal('0.0')

        returns = self.log["實現損益"].dropna()
        total_profit = sum((r for r in returns if r > Decimal('0')), Decimal('0'))
        total_loss = sum((r for r in returns if r < Decimal('0')), Decimal('0')).copy_abs()

        if total_loss == Decimal('0'):
            if total_profit == Decimal('0'):
                return Decimal('0.0')
            return Decimal('Infinity')

        return total_profit / total_loss

    def sortino_ratio(self):
        """計算年化索提諾比率"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return 0.0

        log_df['時間'] = pd.to_datetime(log_df['時間'])
        log_df.set_index('時間', inplace=True)
        daily_returns = log_df['實現損益'].astype(float).resample('D').sum()

        if len(daily_returns) < 2:
            return 0.0

        rf = 0.01 
        mean_daily_return = daily_returns.mean()
        
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return 0.0

        daily_rf = rf / 252
        sortino = (mean_daily_return - daily_rf) / downside_std * np.sqrt(252)
        return sortino

    def calmar_ratio(self):
        """計算卡瑪比率"""
        if "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0

        # 將 pnl 轉為 float，config 中的 initial_cash 也是 float
        total_pnl_float = float(self.pnl)
        initial_cash_float = float(self.config["回測設定"]["initial_cash"])

        if initial_cash_float == 0:
            return 0.0
        
        # 總報酬率
        total_return_rate = total_pnl_float / initial_cash_float
        
        log_df = self.log.copy()
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        if len(log_df) < 2:
             return 0.0
             
        start_date = log_df['時間'].min()
        end_date = log_df['時間'].max()
        num_days = (end_date - start_date).days
        
        if num_days == 0:
            return 0.0

        annualized_return = (1 + total_return_rate) ** (365.0 / num_days) - 1

        max_dd_value = abs(float(self.max_drawdown))
        if max_dd_value == 0:
            if annualized_return > 0:
                return float('inf')
            else:
                return 0.0

        max_dd_percent = max_dd_value / initial_cash_float
        return annualized_return / max_dd_percent

if __name__ == "__main__":
    config = load_config()
    pathdir = "result/backtests"
    filename = f"{config['基本設定']['symbol']}_{config['基本設定']['strategy']}_{config['回測設定']['start_time']} to {config['回測設定']['end_time']}.csv"
    filename = re.sub(":","-",filename)
    df = get_processed_data(filename)

    bt = backtest(df, config)
    bt.run()
    bt.show()
    bt.plot_results()