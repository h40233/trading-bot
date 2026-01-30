# =========================================================================================
# 檔案名稱：backtest.py
# 檔案說明：
#   本檔案實作了「回測引擎 (Backtesting Engine)」的核心邏輯。
#   其運作模式為「事件驅動 (Event-Driven)」的模擬器，透過逐行迭代 K 線資料 (DataFrame)，
#   模擬真實市場中的時間流逝。
#
# 主要職責：
#   1. 環境模擬：讀取歷史數據，還原當時的價格與時間情境。
#   2. 訂單執行：模擬交易所的搓合邏輯，包含滑價 (Slippage)、手續費 (Fee) 與保證金檢查。
#   3. 狀態管理：維護當前的持倉 (Position) 狀態與帳戶資金 (Equity)。
#   4. 績效統計：記錄每一筆交易的詳細數據，並計算夏普比率、最大回撤等關鍵指標。
#
# 架構組成：
#   - class backtest: 主控制器，負責協調數據流與邏輯判斷。
#   - class position: 倉位管理器，專注於計算均價、損益與平倉邏輯。
#   - class stats:    會計模組，負責記帳與生成績效報表。
# =========================================================================================

# [模組引用解析]
# 引用專案內部的工具函式，包含設定讀取與資料處理工具。
from util import *
# 使用 logging 模組進行結構化的日誌輸出，便於除錯與追蹤交易流程。
import logging
# 引入數學模組，主要用於浮點數比對 (math.isclose) 以避免精度誤差。
import math
# 引入繪圖庫，用於最後繪製資金曲線圖。
import matplotlib.pyplot as plt
# 引入 NumPy 進行數值運算，特別是在將 DataFrame 轉換為陣列以加速迭代時使用。
import numpy as np
# 引入進度條工具，讓使用者在 CLI 介面下能看見回測執行進度。
from tqdm import tqdm
# 引入 Pandas 處理結構化的 K 線資料與交易日誌表格。
import pandas as pd

# 設定全域日誌等級為 INFO，確保關鍵交易資訊能被輸出到終端機。
logging.basicConfig(level=logging.INFO)

# =========================================================================================
# 類別：backtest
# 職責：回測主控制器 (Controller)
# 說明：
#   負責接收 K 線資料與設定檔，並執行主要的 `run` 迴圈。
#   它不直接處理數學計算，而是調用 `position` 與 `stats` 物件來完成具體工作。
# =========================================================================================
class backtest:
    """只負責呼叫其他方法"""
    
    # [方法：初始化]
    # 輸入：
    #   - df: 包含歷史 K 線數據與策略信號的 DataFrame。
    #   - config: 系統設定檔 (Dictionary)。
    # 邏輯：
    #   初始化必要的成員變數，並實例化倉位管理 (position) 與統計 (stats) 物件。
    def __init__(self, df, config):
        self.df = df
        self.config = config
        
        # [變數追蹤] self.max_hold
        # 意義：最大持倉 K 棒數限制。若設為 None 代表不限制。
        # 動機：從 config 中讀取並轉型，確保後續邏輯判斷類型正確。
        self.max_hold = config["基本設定"].get("max_hold", None)
        if self.max_hold is not None:
            self.max_hold = int(self.max_hold)

        # 實例化倉位管理器，負責處理開平倉邏輯。
        self.position = position(self.config)
        # 實例化統計模組，負責記錄損益。
        self.stats = stats(self.config)

    # [方法：創建訂單]
    # 輸入：
    #   - close: 當前 K 線收盤價。
    #   - direction: 交易方向 (1 為多，-1 為空)。
    #   - timestamp: 當前時間戳。
    #   - i: 當前資料的索引值 (Index)。
    # 邏輯：
    #   1. 計算止盈 (TP) 與止損 (SL) 價格。
    #   2. 根據設定 (固定金額、百分比或固定數量) 計算下單部位大小 (Size)。
    #   3. 檢查帳戶餘額是否足夠支付保證金與手續費。
    #   4. 呼叫 self.position.open 執行開倉。
    def _create_order(self, close: float, direction: int, timestamp, i: int):
        """內部方法，根據方向創建訂單"""
        
        # [控制流解析] 參數驗證
        # 確保方向參數只有 1 或 -1，防止邏輯錯誤。
        if direction not in [1, -1]:
            raise ValueError("direction 必須是 1 (多) 或 -1 (空)")

        # --- 計算止盈 (TP) ---
        tp_value = float(self.config["止盈止損設定"]["tp_value"])
        # [控制流解析] TP 計算模式
        # 若為百分比模式，則以當前價格為基準按比例計算；否則直接加減點數。
        if self.config["止盈止損設定"]["tp_of_percent"]:
            tp = close * (1 + direction * tp_value / 100)
        else:
            tp = close + direction * tp_value

        # --- 計算止損 (SL) ---
        sl_value = float(self.config["止盈止損設定"]["sl_value"])
        # [控制流解析] SL 計算模式
        # 同上，區分百分比與固定點數模式。
        if self.config["止盈止損設定"]["sl_of_percent"]:
            sl = close * (1 - direction * sl_value / 100)
        else:
            sl = close - direction * sl_value
        
        # --- 計算下單數量 (Size) ---
        order_val = float(self.config["下單設定"]["order_value"])
        
        # [控制流解析] 倉位大小計算模式
        # - percent: 使用當前現金餘額的百分比來決定投入金額。
        # - price: 指定固定的投入金額 (USDT)。
        # - fixed: 指定固定的幣種數量 (如 0.1 BTC)。 # 註1
        if self.config["下單設定"]["order_mode"] == "percent":
            base_size = self.stats.cash * order_val / 100 / close
        elif self.config["下單設定"]["order_mode"] == "price":
            base_size = order_val / close
        elif self.config["下單設定"]["order_mode"] == "fixed":
            base_size = order_val
        else:
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")

        # 讀取槓桿倍數，並計算最終下單數量 (含方向)。
        leverage = float(self.config["下單設定"]["leverage"])
        size = float(direction) * base_size * leverage

        # --- 資金不足判斷 ---
        # 計算所需保證金 (Margin) 與預估手續費。
        margin = close * base_size
        estimated_fee = margin * leverage * self.position.fee_rate
        
        # [控制流解析] 餘額檢查
        # 若現金不足以支付保證金加手續費，則強制放棄該次開倉機會並記錄警告。
        if self.stats.cash < (margin + estimated_fee):
            logging.warning(f"[INSUFFICIENT_FUNDS] 時間: {timestamp}, 資金 {self.stats.cash:.2f} 不足，無法開倉 (需要 {margin+estimated_fee:.2f})")
            return 

        # 呼叫 Position 物件執行開倉，並接收回傳的交易日誌 (若有觸發反向平倉等情況)。
        logs_to_process = self.position.open(close, size, tp, sl, timestamp, i)

        # 若有產生交易日誌 (例如反手時先平倉的損益)，則寫入統計模組。
        if logs_to_process:
            for pnl, log_event in logs_to_process:
                self.stats.trade_log(pnl, log_event)

    # [方法：顯示結果]
    # 邏輯：將 stats 物件計算好的各項績效指標輸出到 Log。
    def show(self):
        logging.info(f"===== 績效總結 =====")
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl:.2f}, 最終資金: {self.stats.cash:.2f}, 最大回撤: {self.stats.max_drawdown:.2f}")
        logging.info(f"勝率 (總/多/空): {self.stats.winrate():.2f}% / {self.stats.long_winrate():.2f}% / {self.stats.short_winrate():.2f}%")
        logging.info(f"獲利因子: {self.stats.profit_factor():.2f}, 夏普比率: {self.stats.sharpe():.2f}, 索提諾比率: {self.stats.sortino_ratio():.2f}, 卡瑪比率: {self.stats.calmar_ratio():.2f}")

    # [方法：繪製圖表]
    # 邏輯：呼叫 stats 物件的繪圖功能，生成資金曲線圖。
    def plot_results(self):
        """繪製並儲存資金曲線圖"""
        self.stats.plot_equity_curve()

    # [方法：執行回測]
    # 輸入：
    #   - progress_callback: 選用參數，用於外部 UI (如 Streamlit) 顯示進度條。
    # 邏輯：
    #   這是回測的主迴圈，負責遍歷每一根 K 線，檢查止盈止損與策略信號。
    def run(self, progress_callback=None):
        # 防呆檢查：若資料為空則直接退出。
        if self.df.empty:
            logging.info("DataFrame is empty, skipping backtest.")
            return

        # --- 將 DataFrame 欄位轉為 NumPy 陣列以加速存取 ---
        # [優化] 使用 NumPy Array 進行迭代比直接操作 DataFrame iterrows 快上數十倍。
        closes = self.df['close'].to_numpy(dtype=float)
        signals = self.df['signal'].to_numpy(dtype=np.int8)
        close_times = self.df['close_time'].to_numpy()
        
        total_len = len(self.df)
        
        # [迴圈] 遍歷每一筆歷史數據 # 註2
        # for i in tqdm(range(total_len)): # CLI環境下的進度條 (註解掉的備用程式碼)
        for i in range(total_len):
            # 若有提供進度回呼函式，則每 1% 更新一次進度。
            if progress_callback and (i % (total_len // 100 + 1) == 0):
                progress_callback(i / total_len)

            # [安全機制] 破產保護
            # 若現金歸零或為負，強制終止回測。
            if self.stats.cash <= 0:
                logging.info("資金不足，無法繼續交易")
                break
            
            # 取得當前的市場狀態 (收盤價與時間)。
            current_close = closes[i]
            current_time = close_times[i]
            
            # [狀態判斷] 檢查是否有持倉
            if self.position.size != 0:
                # 1. 檢查止損 (Stop Loss)
                # 若當前價格觸發 SL，position 物件會回傳平倉日誌。
                logs_sl = self.position.trigger_SL(current_close, current_time)
                if logs_sl:
                    for pnl, log_event in logs_sl:
                        self.stats.trade_log(pnl, log_event)

                # 2. 檢查持倉時間上限 (Time-based Exit)
                # 若設定了 max_hold 且持倉時間超過設定值，強制平倉。
                if self.max_hold is not None and (i - self.position.entry_index) >= self.max_hold:
                    logs_force_close = self.position.close_all(current_close, current_time)
                    if logs_force_close:
                        for pnl, log_event in logs_force_close:
                            self.stats.trade_log(pnl, log_event)

                # 3. 檢查止盈 (Take Profit)
                # 若當前價格觸發 TP，執行平倉。
                logs_tp = self.position.trigger_TP(current_close, current_time)
                if logs_tp:
                    for pnl, log_event in logs_tp:
                        self.stats.trade_log(pnl, log_event)
            
            # 4. 檢查進場信號 (Entry Signal)
            # 讀取策略預先計算好的信號 (1:多, -1:空)。
            current_signal = signals[i]
            if current_signal == 1:
                self._create_order(current_close, 1, current_time, i)
            elif current_signal == -1:
                self._create_order(current_close, -1, current_time, i)
        
        # --- 迴圈結束後處理 ---
        # 回測結束時若還有持倉，以最後一根 K 線的價格強制平倉，以計算最終淨值。
        if self.position.size != 0:
            last_price = closes[-1]
            last_time = close_times[-1]
            close_results = self.position.close_all(last_price, last_time)
            if close_results:
                pnl, log = close_results[0]
                self.stats.trade_log(pnl, log)
        
        # --- 一次性生成 Log DataFrame ---
        # 將列表形式的日誌轉換為 Pandas DataFrame，方便後續分析。
        self.stats.finalize_log()

        # 更新進度條至 100%。
        if progress_callback:
            progress_callback(1.0)
        
        # 輸出最終摘要。
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl:.2f}, 最大回撤: {self.stats.max_drawdown:.2f}, 最終資金: {self.stats.cash:.2f}")
        
        # 若有交易紀錄，則儲存為 CSV 檔案。
        if not self.stats.log.empty:
            result_to_csv(self.stats.log, is_backtest=True)

# =========================================================================================
# 類別：position
# 職責：管理單一倉位 (Position) 的狀態
# 說明：
#   這是一個狀態機 (State Machine)，維護著當前的持倉量、均價、TP/SL 設定。
#   它負責計算「開倉後的平均價格」以及「平倉時的已實現損益」。
# =========================================================================================
class position:
    """只負責倉位部分的動作"""
    
    # [方法：初始化]
    # 邏輯：重置所有狀態變數，並讀取費率與滑價設定。
    def __init__(self, config):
        self.config = config
        self.avg_price = 0.0 # 持倉均價 (加權平均)
        self.size = 0.0      # 持倉數量 (正數為多，負數為空)
        self.tp = 0.0        # 止盈價格
        self.sl = 0.0        # 止損價格
        self.entry_index = 0 # 進場時的 K 線 Index (用於計算持倉時間)
        self.fee_rate = float(config["回測設定"]["fee_rate"]) # 手續費率 (如 0.0004)
        self.slippage = float(config["回測設定"]["slippage"]) # 滑價率 (如 0.0005)
        self.allow_pyramiding = bool(config["下單設定"]["pyramiding"]) # 是否允許加倉
        self.allow_reverse = bool(config["下單設定"]["reverse"])       # 是否允許反手

    # [方法：開倉]
    # 輸入：
    #   - price: 觸發開倉的基準價格 (通常是 Close)。
    #   - size: 欲開倉的數量 (含正負號)。
    # 回傳：包含 (損益, 日誌字典) 的列表。
    def open(self, price: float, size: float, tp: float, sl: float, timestamp, entry_index: int) -> list[tuple[float, dict]]:
        """開倉，回傳 (pnl, log_dict) 的列表"""
        
        # 檢查下單量是否過小 (浮點數誤差處理)。
        if abs(size) < 1e-12: # 檢查 size 是否趨近於 0
            raise ValueError("size不能為0")

        direction = 1 if size > 0 else -1
        
        # [邏輯] 滑價處理與價格驗證
        # 模擬真實市場：買入價會比現價高 (滑價向上)，賣出價會比現價低 (滑價向下)。
        if direction == 1: # 做多
            # 檢查 TP/SL 邏輯合理性：多單應為 SL < Price < TP。
            if not (sl is None or tp is None or sl < price < tp):
                 if sl is not None and tp is not None and not (sl < price < tp):
                    raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
            price *= (1 + self.slippage)
        else: # 做空
            # 檢查 TP/SL 邏輯合理性：空單應為 TP < Price < SL。
            if not (sl is None or tp is None or tp < price < sl):
                if sl is not None and tp is not None and not (tp < price < sl):
                    raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
            price *= (1 - self.slippage)
        
        # [邏輯] 加倉與反手判斷
        if self.size != 0:
            # 情況 A: 反向開單 (例如當前有多單，但訊號做空)
            if self.size * size < 0:
                if self.allow_reverse:
                    # 若允許反手，則先平掉舊倉位，再開新倉位。
                    return self.reverse(price, size, tp, sl, timestamp, entry_index)
                return [] # 若不允許反手，則忽略此訊號。
            # 情況 B: 同向開單 (加倉)
            else:
                if not self.allow_pyramiding:
                    return [] # 若不允許加倉，則忽略。
        
        # [核心算法] 計算新的加權平均價格 (Average Entry Price)
        # 公式：(舊均價 * 舊量 + 新價 * 新量) / (舊量 + 新量)
        self.avg_price = (self.avg_price * abs(self.size) + price * abs(size)) / (abs(self.size) + abs(size))
        self.size += size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index 

        # 建立開倉日誌
        log = {
            "時間": timestamp, "狀態": "開倉", "多/空": direction, 
            "進場價": price, "進場量": size, "當前均價": self.avg_price, "當前持倉量": self.size
        }
        # 開倉當下尚未實現損益，故 PnL 為 0.0。
        return [(0.0, log)] 

    # [方法：平倉]
    # 輸入：
    #   - price: 平倉基準價。
    #   - size_to_close: 欲平倉的數量 (需與持倉方向相反)。
    # 邏輯：計算已實現損益 (Realized PnL) 並扣除手續費。
    def close(self, price: float, size_to_close: float, timestamp) -> list[tuple[float, dict]]:
        """平倉，回傳 (pnl, log_dict) 的列表"""
        if self.size == 0:
            raise Exception("當前無持倉，無法平倉")

        direction_to_close = 1 if size_to_close > 0 else -1
        
        # 根據滑價調整實際成交價 (平倉時同樣會有滑價成本)。
        if direction_to_close == 1: # 買入平倉(空單)
            price *= (1 + self.slippage) 
        else: # 賣出平倉(多單)
            price *= (1 - self.slippage) 

        
        # --- 根據多空方向，使用不同的損益計算邏輯 ---
        if self.size > 0: # 原多單，賣出平倉
            gross_pnl = (price - self.avg_price) * abs(size_to_close)
        else: # 原空單，買入平倉
            gross_pnl = (self.avg_price - price) * abs(size_to_close)
            
        # 計算平倉手續費
        closing_fee = abs(size_to_close) * price * self.fee_rate
        # 淨損益 = 毛損益 - 手續費
        pnl = gross_pnl - closing_fee
        
        # 更新剩餘持倉量
        self.size += size_to_close
        
        # 浮點數歸零修正：若剩餘量極小，視為完全平倉。
        if abs(self.size) < 1e-9: # 避免浮點數問題
            self.size = 0.0
            self.avg_price = 0.0
            
        log = {
            "時間": timestamp, "狀態": "平倉", "出場價": price, 
            "出場量": size_to_close, "實現損益": pnl, "剩餘倉位": self.size
        }
        return [(pnl, log)]
    
    # [方法：全平]
    # 邏輯：將當前所有持倉量一次性平掉。
    def close_all(self, price: float, timestamp) -> list[tuple[float, dict]]:
        """全部平倉"""
        if self.size == 0:
            return []
        # 平倉量 = 當前持倉量的負值
        size_to_close = -self.size
        # 重置 TP/SL 設定
        self.sl = None
        self.tp = None
        return self.close(price, size_to_close, timestamp)

    # [方法：反手]
    # 邏輯：先執行全平 (Close All)，再執行開倉 (Open)。
    def reverse(self, price: float, new_size: float, tp: float, sl: float, timestamp, entry_index:int) -> list[tuple[float, dict]]:
        """反手"""
        # 步驟 1: 平掉舊倉
        close_results = self.close_all(price, timestamp)
        if not close_results:
            return []
        
        pnl_close, log_close = close_results[0]
        
        # 步驟 2: 開新倉
        direction = 1 if new_size > 0 else -1
        self.avg_price = price
        self.size = new_size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index
        
        log_open = {
            "時間": timestamp, "狀態": "開倉", "多/空": direction, "進場價": price, 
            "進場量": new_size, "當前均價": self.avg_price, "當前持倉量": self.size
        }
        
        # 回傳兩筆紀錄：一筆平倉損益，一筆開倉紀錄。
        return [(pnl_close, log_close), (0.0, log_open)]

    # [方法：觸發止損]
    # 邏輯：檢查當前價格是否觸及 SL 水位。
    def trigger_SL(self, close: float, timestamp):
        if self.sl is not None and self.sl != 0:
            # 使用 isclose 來處理浮點數精度問題
            is_close = math.isclose(close, self.sl)
            # 多單：價格 <= SL；空單：價格 >= SL
            if (self.size > 0 and (close <= self.sl or is_close)) or \
               (self.size < 0 and (close >= self.sl or is_close)):
                return self.close_all(self.sl, timestamp)
        return []
    
    # [方法：觸發止盈]
    # 邏輯：檢查當前價格是否觸及 TP 水位。
    def trigger_TP(self, close: float, timestamp):
        if self.tp is not None and self.tp != 0:
            # 使用 isclose 來處理浮點數精度問題
            is_close = math.isclose(close, self.tp)
            # 多單：價格 >= TP；空單：價格 <= TP
            if (self.size > 0 and (close >= self.tp or is_close)) or \
               (self.size < 0 and (close <= self.tp or is_close)):
                return self.close_all(self.tp, timestamp)
        return []

# =========================================================================================
# 類別：stats
# 職責：統計與會計模組 (Statistics)
# 說明：
#   負責收集來自 backtest 的交易事件，更新現金餘額 (Cash Balance)，
#   並計算各類金融績效指標 (Win Rate, Sharpe, Drawdown)。
# =========================================================================================
class stats:
    """只負責記錄資料的動作"""
    
    # [方法：初始化]
    # 邏輯：初始化各項計數器與資金狀態。
    def __init__(self, config):
        self.config = config
        self.count = 0          # 總交易次數
        self.count_long = 0     # 做多交易次數
        self.count_long_win = 0 # 做多獲利次數
        self.count_short = 0    # 做空交易次數
        self.count_short_win = 0# 做空獲利次數
        
        self.log_events = [] # 儲存交易日誌的列表 (List of Dicts)
        self.log = pd.DataFrame() # 最終輸出的 DataFrame
        
        self.cash = float(config["回測設定"]["initial_cash"]) # 當前可用資金
        self.pnl = 0.0          # 累計損益
        self.peak_equity = self.cash # 用於計算回撤的權益峰值
        self.max_drawdown = 0.0 # 最大回撤金額

    # [方法：紀錄交易日誌]
    # 輸入：
    #   - pnl: 該筆交易的損益 (開倉時通常為 0 或負手續費)。
    #   - log_event: 交易詳情字典。
    # 邏輯：根據開倉或平倉狀態，更新現金水位與統計數據。
    def trade_log(self, pnl: float, log_event: dict):
        if log_event is not None:
            self.log_events.append(log_event)

            # 處理開倉事件：主要扣除開倉手續費。
            if log_event.get("狀態") == "開倉":
                fee_rate = float(self.config["回測設定"]["fee_rate"])
                opening_fee = abs(log_event["進場量"]) * log_event["進場價"] * fee_rate
                self.cash -= opening_fee
                self.pnl -= opening_fee

            # 處理平倉事件：結算損益並更新統計指標。
            elif log_event.get("狀態") == "平倉":
                if pnl is None:
                    return

                self.cash += pnl
                # 更新 PnL & Drawdown
                current_equity = self.cash
                self.pnl += pnl # 這裡的pnl是已經扣掉手續費的
                
                # 更新最大權益峰值，用於計算回撤 (DD)
                self.peak_equity = max(self.peak_equity, current_equity)
                # 當前權益與峰值的差距
                drawdown = self.peak_equity - current_equity
                self.max_drawdown = max(self.max_drawdown, drawdown)

                closed_size = log_event["出場量"]
                self.count += 1
                
                # [邏輯] 勝率統計分類
                # 根據出場量正負判斷原單方向
                if closed_size > 0: # 買入平倉 (原空單)
                    self.count_short += 1
                    if pnl > 0: self.count_short_win += 1
                else: # 賣出平倉 (原多單)
                    self.count_long += 1
                    if pnl > 0: self.count_long_win += 1

    # [方法：生成最終日誌]
    # 說明：將 list 轉換為 DataFrame，提升後續查詢效率。
    def finalize_log(self):
        """在回測結束後，一次性生成最終的 DataFrame。"""
        if self.log_events:
            self.log = pd.DataFrame(self.log_events)
    
    # [方法：夏普比率計算]
    # 說明：衡量每單位風險所獲得的超額報酬。
    # 邏輯：(平均日報酬 - 無風險利率) / 日報酬標準差 * sqrt(252)。
    def sharpe(self):
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0 

        temp_log = self.log.copy()
        temp_log['時間'] = pd.to_datetime(temp_log['時間'])
        temp_log.set_index('時間', inplace=True)

        # 將交易損益重取樣 (Resample) 為日別損益。
        daily_returns = temp_log['實現損益'].resample('D').sum()

        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return 0.0

        rf_per_day = 0.01 / 252 # 假設年無風險利率為 1%
        avg_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        sharpe_ratio = (avg_return - rf_per_day) / std_return * np.sqrt(252)
        return sharpe_ratio

    # [方法：獲取資金曲線數據]
    # 說明：計算用於繪圖的時間序列數據，包含淨值 (Equity) 與回撤 (Drawdown) 百分比。
    def get_equity_curve(self):
        """
        計算資金曲線與水下圖數據。
        返回一個包含 '時間', '資金曲線', '回撤' 的 DataFrame。
        """
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].dropna().empty:
            return None

        pnl_events = self.log[self.log['實現損益'].notna()].copy()
        pnl_events['時間'] = pd.to_datetime(pnl_events['時間'])
        pnl_events = pnl_events.sort_values(by='時間')
        
        initial_cash = float(self.config["回測設定"]["initial_cash"])
        # 使用 cumsum() 計算累計損益
        pnl_events['累計損益'] = pnl_events['實現損益'].cumsum()
        pnl_events['資金曲線'] = initial_cash + pnl_events['累計損益']
        
        # 計算水下圖 (Drawdown)
        # 滾動高點：目前為止出現過的最高資金水位
        pnl_events['滾動高點'] = pnl_events['資金曲線'].expanding().max()
        # 計算回撤百分比公式：(當前資金 - 最高資金) / 最高資金
        pnl_events['回撤'] = (pnl_events['資金曲線'] - pnl_events['滾動高點']) / pnl_events['滾動高點']
        
        return pnl_events[['時間', '資金曲線', '回撤']]

    # [方法：獲取週期性回報]
    # 說明：計算月/季/年的匯總損益，用於 Bar Chart 顯示。
    def get_periodic_returns(self, period='M'):
        """
        計算週期性報酬。
        :param period: 'M' for Monthly, 'A' for Annual, 'Q' for Quarterly
        :return: A DataFrame with periodic returns.
        """
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].dropna().empty:
            return None
        
        pnl_log = self.log[self.log['實現損益'].notna()].copy()
        pnl_log['時間'] = pd.to_datetime(pnl_log['時間'])
        pnl_log = pnl_log.set_index('時間')
        
        periodic_returns = pnl_log['實現損益'].resample(period).sum()
        
        # 轉換為 DataFrame 並重設索引，方便 Plotly 處理
        periodic_returns = periodic_returns.reset_index()
        periodic_returns.columns = ['時間', '損益']
        
        return periodic_returns

    # [方法：繪製資金曲線]
    # 說明：使用 Matplotlib 繪製簡單的靜態圖表 (主要用於 CLI 模式)。
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

    # [輔助方法：計算勝率]
    def long_winrate(self):
        if self.count_long == 0: return 0.0
        return (self.count_long_win / self.count_long) * 100

    def short_winrate(self):
        if self.count_short == 0: return 0.0
        return (self.count_short_win / self.count_short) * 100

    def winrate(self):
        if self.count == 0: return 0.0
        return ((self.count_long_win + self.count_short_win) / self.count) * 100

    # [方法：獲利因子]
    # 公式：總獲利金額 / 總虧損金額。
    def profit_factor(self):
        """計算獲利因子"""
        if self.log.empty or "實現損益" not in self.log.columns:
            return 0.0

        returns = self.log["實現損益"].dropna()
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())

        if total_loss == 0:
            return np.inf if total_profit > 0 else 0.0
        return total_profit / total_loss

    # [方法：索提諾比率]
    # 說明：類似夏普比率，但分母只計算「下行風險」(負報酬的標準差)。
    def sortino_ratio(self):
        """計算年化索提諾比率"""
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0

        temp_log = self.log.copy()
        temp_log['時間'] = pd.to_datetime(temp_log['時間'])
        temp_log.set_index('時間', inplace=True)
        daily_returns = temp_log['實現損益'].resample('D').sum()

        if len(daily_returns) < 2: return 0.0

        rf_per_day = 0.01 / 252
        mean_daily_return = daily_returns.mean()
        
        # 只取負報酬的部分來計算標準差
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return 0.0

        sortino = (mean_daily_return - rf_per_day) / downside_std * np.sqrt(252)
        return sortino

    # [方法：卡瑪比率]
    # 說明：衡量收益與最大回撤的關係。公式：年化報酬率 / 最大回撤率。
    def calmar_ratio(self):
        """計算卡瑪比率"""
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0
        
        initial_cash = float(self.config["回測設定"]["initial_cash"])
        if initial_cash == 0: return 0.0
        
        # 計算年化報酬率
        temp_log = self.log.copy()
        temp_log['時間'] = pd.to_datetime(temp_log['時間'])
        start_date = temp_log['時間'].min()
        end_date = temp_log['時間'].max()
        num_days = (end_date - start_date).days
        if num_days < 1: return 0.0

        total_return_rate = self.pnl / initial_cash
        annualized_return = (1 + total_return_rate) ** (365.0 / num_days) - 1

        # 最大回撤值 (這裡是負數，取絕對值)
        max_dd_value = abs(self.max_drawdown)
        if max_dd_value == 0:
            return np.inf if annualized_return > 0 else 0.0

        return annualized_return / (max_dd_value / initial_cash)

# [程式進入點]
# 說明：當直接執行此檔案 (python backtest.py) 時會運行的測試區塊。
if __name__ == "__main__":
    config = load_config()
    # 確保 if __name__ == "__main__": 下的路徑和檔案名稱生成邏輯正確
    import re
    # 檔案名稱中的時間格式可能包含不適合做檔名的 ":" 符號，需要替換
    start_time_str = re.sub(":", "-", config['回測設定']['start_time'])
    end_time_str = re.sub(":", "-", config['回測設定']['end_time'])
    # 從 util.py 取得 get_processed_data 函式 (假設它存在)
    df = get_processed_data(f"{config['基本設定']['symbol']}_{config['基本設定']['timeframe']}_{config['基本設定']['strategy']}_{start_time_str} to {end_time_str}.csv")

    if df is not None:
        bt = backtest(df, config)
        bt.run()
        bt.show()
        bt.plot_results()
    else:
        print("找不到對應的 Processed Data，請先執行一次回測來產生。")

# ====== 備註區 ======
# 註1: 下單模式的擴展性建議
#      目前的 fixed 模式是固定「數量」。在加密貨幣交易中，通常會希望固定「U本位金額」進行複利。
#      建議檢查 'price' 模式是否即為固定金額模式，若是，命名可改為 'fixed_cash' 更直觀。
#
# 註2: 回測效能優化 (Vectorization)
#      目前的 run() 方法使用 for 迴圈遍歷每一根 K 線，這在 Python 中效率較低。
#      若策略邏輯不包含複雜的路徑依賴 (如追蹤止損動態變化)，建議改用 VectorBT 或自行實作向量化回測，
#      將迴圈改為 NumPy 陣列運算，速度可提升 100 倍以上。