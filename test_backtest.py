# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是一個完整的測試套件，用於驗證 `backtest.py` 模組的所有功能。
# 測試範圍包括：
# 1. Position Class: 開倉、平倉、反手、加倉 (Pyramiding)、損益計算。
# 2. Stats Class: 交易紀錄統計、勝率、獲利因子、最大回撤 (Max Drawdown)。
# 3. Backtest Class (整合測試): 模擬完整的回測流程，驗證最終資金與交易次數。
# 4. Edge Cases: 測試資金不足、空資料、錯誤參數等異常狀況的處理能力。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 引入 pytest，測試框架的核心。
import pytest

# 引入 pandas，用於建立測試用的 K 線資料 (DataFrame)。
import pandas as pd

# 引入 numpy，科學計算庫 (雖然這裡主要用 Decimal，但 pandas 底層依賴它)。
import numpy as np

# 引入 math，主要用於浮點數比較 (math.isclose)，因為小數點運算常有微小誤差。
import math

# 引入 Decimal，這是高精度的數字型態，專門用於金融計算，避免浮點數誤差 (如 0.1 + 0.2 != 0.3)。
from decimal import Decimal

# 引入 logging，用於測試日誌輸出功能 (如 caplog)。
import logging

# 從 unittest.mock 引入 patch，用於模擬外部依賴 (如畫圖、存檔)。
from unittest.mock import patch

# 引入被測試的模組與類別 (假設 backtest.py 在同一個目錄下)。
# position: 負責倉位管理。
# stats: 負責統計績效。
# backtest: 回測引擎主程式。
from backtest import position, stats, backtest

# [Fixture 說明]
# 功能：建立一個簡單的 K 線資料表，供多個測試案例共用。
# 原理：包含時間、收盤價、和預設的交易訊號 (1:買, 0:不變, -1:賣)。
@pytest.fixture
def sample_df():
    """提供一個用於回測的樣本 DataFrame"""
    data = {
        # 時間需要是 datetime 物件，以便 backtest 內部做 resample 等時間操作。
        'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00', '2024-01-01 03:00', '2024-01-01 04:00']),
        'close': [100, 110, 95, 105], # 模擬價格波動
        'signal': [1, 0, -1, 0]  # 買入 -> 持有 -> 反手做空 -> 持有
    }
    return pd.DataFrame(data)

# [Fixture 說明]
# 功能：模擬 config.json 的內容。
# 原理：直接回傳一個 Python 字典，讓測試程式不需要依賴外部檔案，保持獨立性。
@pytest.fixture
def mock_config():
    """模擬 config，避免測試依賴於真實的 config.json 檔案"""
    config_data = {
        "基本設定": {"symbol": "TESTUSDT", "strategy": "TEST", "max_hold": None}, # 預設不啟用 max_hold
        "下單設定": {"order_mode": "fixed", "order_value": 1, "leverage": 1, "pyramiding": False, "reverse": True},
        "止盈止損設定": {"tp_of_percent": True, "tp_value": 10, "sl_of_percent": True, "sl_value": 5},
        "回測設定": {"initial_cash": 10000, "fee_rate": 0.001, "slippage": 0.0} # 測試時滑價設為0，方便計算
    }
    return config_data

# [Class 說明]
# 職責：測試 `position` 類別的所有方法。
# 包含：開倉 (open)、平倉 (close)、反手 (reverse)、加倉 (pyramiding)。
class TestPosition:
    """測試 position class 的核心功能"""
    
    # [Function 說明]
    # 功能：測試開多單 (Long) 的邏輯。
    # 原理：呼叫 pos.open，驗證倉位大小、均價、以及回傳的日誌是否正確。
    def test_open_long(self, mock_config):
        # 初始化 position 物件。
        pos = position(mock_config)
        # open 現在回傳一個列表 (因為可能有多次操作，如反手時會先平再開)。
        # 這裡模擬在價格 100 開倉 1 單位。
        results = pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        pnl, log = results[0] # 取出第一筆操作結果
        
        # 驗證開倉當下不應該有實現損益 (PnL = 0)。
        assert pnl == Decimal('0')
        # 驗證持倉量更新為 1。
        assert pos.size == Decimal('1')
        # 驗證持倉均價更新為 100。
        assert pos.avg_price == Decimal('100')
        # 驗證日誌中有紀錄 "開倉"。
        assert "開倉" in log["狀態"].values

    # [Function 說明]
    # 功能：測試多單獲利平倉的邏輯與損益計算。
    # 原理：先開倉，再平倉，手動計算 PnL 並與程式結果比對。
    def test_close_long_profit(self, mock_config):
        pos = position(mock_config)
        # 先在 100 元買入 1 單位。
        pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        
        # 在 105 元平倉 (size=-1 代表賣出)。
        results = pos.close(price=Decimal('105'), size=Decimal('-1'), timestamp='2024-01-01 02:00')
        pnl, log = results[0]
        
        # 新邏輯: PnL = 毛利 - 平倉手續費
        # 毛利 = (賣價 105 - 買價 100) * 數量 1 = 5
        # 手續費 = 交易金額 105 * 費率 0.001 = 0.105
        # PnL = 5 - 0.105 = 4.895
        assert math.isclose(pnl, Decimal('4.895'))
        # 驗證倉位歸零。
        assert pos.size == Decimal('0')
        # 驗證日誌狀態。
        assert "平倉" in log["狀態"].values

    # [Function 說明]
    # 功能：測試反手策略 (Reverse)，即從多單直接轉為空單。
    # 原理：系統會先平掉多單，再開空單。這裡主要測試平倉部分的損益計算。
    def test_reverse_to_short(self, mock_config):
        pos = position(mock_config)
        # 先在 100 元買入。
        pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        
        # 由於 sl=95，當價格到達95時，會觸發止損，所以平倉價是95。
        # reverse 現在回傳一個包含 [平倉結果, 開倉結果] 的列表。
        results = pos.reverse(price=Decimal('95'), size=Decimal('-1'), tp=Decimal('85.5'), sl=Decimal('99.75'), timestamp='2024-01-01 03:00', entry_index=1)
        
        # 驗證第一步：平倉部分的損益。
        pnl_close, log_close = results[0]
        
        # 新邏輯: PnL = 毛利 - 平倉手續費
        # 毛利 = (賣價 95 - 買價 100) * 1 = -5 (虧損)
        # 手續費 = 95 * 0.001 = 0.095
        # PnL = -5 - 0.095 = -5.095
        assert math.isclose(pnl_close, Decimal('-5.095'))
        # 驗證最終倉位變成 -1 (空單)。
        assert pos.size == Decimal('-1') 
        # 驗證均價更新為新的開倉價 95。
        assert math.isclose(pos.avg_price, Decimal('95')) 

    # [Function 說明]
    # 功能：測試加倉 (Pyramiding) 功能，即在已有持倉的情況下繼續開倉。
    # 原理：驗證倉位數量疊加，以及均價 (Average Price) 的重新計算。
    def test_pyramiding(self, mock_config):
        """測試加倉功能"""
        # 開啟加倉設定。
        mock_config["下單設定"]["pyramiding"] = True
        pos = position(mock_config)

        # 第一次開倉：100元買1個。
        pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        assert pos.size == Decimal('1')
        assert pos.avg_price == Decimal('100')

        # 第二次加倉：110元買1個。
        pos.open(price=Decimal('110'), size=Decimal('1'), tp=Decimal('120'), sl=Decimal('100'), timestamp='2024-01-01 02:00', entry_index=1)
        # 驗證總持倉量變成 2。
        assert pos.size == Decimal('2') 
        # 驗證新均價：(100*1 + 110*1) / 2 = 105。
        assert math.isclose(pos.avg_price, Decimal('105')) 

        # 第三次部分平倉 (平掉一半，賣出1個)。
        results = pos.close(price=Decimal('120'), size=Decimal('-1'), timestamp='2024-01-01 03:00')
        pnl, log = results[0]
        # 驗證剩餘持倉為 1。
        assert pos.size == Decimal('1') 
        # 新邏輯: PnL = (賣價 120 - 成本 105) * 1 - (120 * 0.001) = 15 - 0.12 = 14.88。
        assert math.isclose(pnl, Decimal('14.88'))

    def test_slippage_impact(self, mock_config):
        """測試滑價是否正確影響成交價格"""
        # 1. 設定 1% 的滑價 (0.01)
        mock_config["回測設定"]["slippage"] = 0.01 
        pos = position(mock_config)

        # --- 測試買入 (做多開倉) ---
        # 預期：成交價應該比市價貴 1%
        # 100 * (1 + 0.01) = 101
        results = pos.open(price=Decimal('100'), size=Decimal('1'), tp=None, sl=None, timestamp='2024-01-01', entry_index=0)
        _, log_open = results[0]
        executed_price_open = log_open["進場價"].iloc[0]
        
        assert math.isclose(executed_price_open, Decimal('101')), f"買入滑價計算錯誤，預期 101，實際 {executed_price_open}"

        # --- 測試賣出 (平多倉) ---
        # 預期：成交價應該比市價便宜 1% (賣得更低)
        # 200 * (1 - 0.01) = 198
        results = pos.close(price=Decimal('200'), size=Decimal('-1'), timestamp='2024-01-02')
        _, log_close = results[0]
        executed_price_close = log_close["出場價"].iloc[0]

        assert math.isclose(executed_price_close, Decimal('198')), f"賣出滑價計算錯誤，預期 198，實際 {executed_price_close}"

        # --- 測試做空 (開空倉) ---
        # 預期：成交價應該比市價便宜 1% (雖然是做空，但開倉是賣出動作，所以價格會更低/更差)
        # 註：這裡要確認 backtest.py 的邏輯定義。
        # 通常做空(Sell)會有滑價導致賣在更低點。
        # 讓我們檢查 backtest.py 的 open 邏輯：
        # else: # 做空
        #    price = price * (Decimal('1') - self.slippage) 
        # 邏輯正確。
        
        pos.size = Decimal('0') # 重置倉位
        results = pos.open(price=Decimal('100'), size=Decimal('-1'), tp=None, sl=None, timestamp='2024-01-03', entry_index=0)
        _, log_short = results[0]
        executed_price_short = log_short["進場價"].iloc[0]
        
        assert math.isclose(executed_price_short, Decimal('99')), f"做空滑價計算錯誤，預期 99，實際 {executed_price_short}"

# [Class 說明]
# 職責：測試 `stats` 類別的統計功能。
# 包含：交易次數、勝率、獲利因子、最大回撤。
class TestStats:
    """測試 stats class 的計算功能"""
    
    # [Function 說明]
    # 功能：測試基本的交易統計指標 (勝率、總損益等)。
    # 原理：手動餵入兩筆交易紀錄 (一賺一賠)，驗證統計結果是否符合預期。
    def test_trade_log_and_metrics(self, mock_config):
        stat = stats(mock_config)
        
        # 模擬一筆獲利的多單交易 (PnL=100)。
        profit_log = pd.DataFrame([{'狀態': '平倉', '出場量': Decimal('-1'), '實現損益': Decimal('100'), '時間': '2024-01-01'}])
        stat.trade_log(pnl=Decimal('100'), log=profit_log)
        
        # 模擬一筆虧損的空單交易 (PnL=-50)。
        loss_log = pd.DataFrame([{'狀態': '平倉', '出場量': Decimal('1'), '實現損益': Decimal('-50'), '時間': '2024-01-02'}])
        stat.trade_log(pnl=Decimal('-50'), log=loss_log)
        
        # 驗證總交易次數為 2。
        assert stat.count == Decimal('2')
        # 驗證多單次數為 1。
        assert stat.count_long == Decimal('1')
        # 驗證多單勝場數為 1。
        assert stat.count_long_win == Decimal('1')
        # 驗證空單次數為 1。
        assert stat.count_short == Decimal('1')
        # 驗證空單勝場數為 0。
        assert stat.count_short_win == Decimal('0')
        # 驗證總損益 = 100 - 50 = 50。
        assert math.isclose(stat.pnl, Decimal('50'))
        # 驗證最終現金 = 10000 (初始) + 50 = 10050。
        assert math.isclose(stat.cash, Decimal('10050'))
        # 驗證勝率 = 1勝 / 2場 = 50.0%。
        assert math.isclose(stat.winrate(), Decimal('50.0'))
        # 驗證獲利因子 = 總獲利 / 總虧損 = 100 / 50 = 2.0。
        assert math.isclose(stat.profit_factor(), Decimal('2.0')) 

    # [Function 說明]
    # 功能：測試最大回撤 (Max Drawdown) 的計算邏輯。
    # 原理：模擬一系列賺賺賠賠的交易，觀察資產曲線的最高點到最低點的跌幅。
    def test_max_drawdown(self, mock_config):
        """測試最大回撤計算的正確性"""
        stat = stats(mock_config)
        # 1. 賺 100。目前權益 10100。峰值 10100。回撤 0。
        stat.trade_log(pnl=Decimal('100'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('100'), '出場量': Decimal('1')}])) 
        assert math.isclose(stat.max_drawdown, Decimal('0'))

        # 2. 賠 200。目前權益 9900。峰值 10100。當前回撤 = 9900 - 10100 = -200。MDD = -200? (註解寫 -100 可能是因為初始資金也是參考點，這裡邏輯取決於 stats 實作細節，假設測試正確)。
        # 修正理解：MDD 通常是指從「歷史最高點」跌下來的幅度。
        # 峰值是 10100 (初始10000+100)。現在變成 9900。跌了 200。
        # 這裡測試代碼期望是 -100，這可能意味著它的 MDD 定義是相對於「初始資金」還是其他邏輯，我們先照著測試邏輯解釋：
        # 假設邏輯：從賺錢後的最高點跌下來的量。
        stat.trade_log(pnl=Decimal('-200'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('-200'), '出場量': Decimal('1')}])) 
        assert math.isclose(stat.max_drawdown, Decimal('-100')) # 這裡的 -100 可能有誤，或是特定定義，一般來說跌幅是 200。但我們遵循不改code原則，解釋這行在驗證回撤。

        # 3. 賺 50。目前權益 9950。峰值仍是 10100。MDD 保持紀錄。
        stat.trade_log(pnl=Decimal('50'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('50'), '出場量': Decimal('1')}])) 
        assert math.isclose(stat.max_drawdown, Decimal('-100'))

        # 4. 賠 100。目前權益 9850。峰值 10100。總跌幅 10100 - 9850 = 250。
        # 這裡 assert 寫 -150。這暗示了此處的回撤計算邏輯可能比較特殊，或者是累計虧損的概念。
        stat.trade_log(pnl=Decimal('-100'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('-100'), '出場量': Decimal('1')}])) 
        assert math.isclose(stat.max_drawdown, Decimal('-150'))

# [Function 說明]
# 功能：整合測試 (Integration Test)。
# 原理：建立 backtest 物件，跑完 `run()`，並驗證最終的統計結果是否符合預期。
@patch('backtest.result_to_csv') # 模擬檔案寫入，避免產生垃圾檔案。
@patch('backtest.plt.show')     # 模擬繪圖顯示，避免跳出視窗卡住測試。
def test_backtest_run_integration(mock_plt_show, mock_result_to_csv, mock_config, sample_df):
    """對 backtest.run() 進行整合測試，驗證完整流程"""
    bt = backtest(sample_df,mock_config)
    bt.run()

    # --- 驗證最終結果 ---
    # 手動計算預期結果：
    # 交易1 (止盈): 100買，110賣。PnL = 10 - 0.1 - 0.11 = 9.79。
    # 交易2 (反手止損): 100買，95賣(觸發止損)。但這裡要注意，sample_df 裡第三個點是 signal -1 (反手)。
    # 讓我們看 sample_df:
    # 01:00 close 100, signal 1 -> 開多。
    # 02:00 close 110, signal 0 -> 觸發 tp=110? 預設 tp=10% (110)。剛好觸發。平倉。
    # 03:00 close 95, signal -1 -> 反手做空。在 95 開空。
    # 04:00 close 105, signal 0 -> 持有。
    # 結束回測 -> 強制平倉。
    
    # 根據測試代碼的註解推算：
    # 交易2 是 "止損"? 註解寫: PnL = (95-99.75)*1 ... 這看起來像是第二筆交易是空單，然後止損了?
    # 讓我們回頭看 mock_config: sl_value=5%。95 的 5% 是 4.75。95+4.75 = 99.75。
    # 所以如果在 95 開空，價格漲到 99.75 就會止損。
    # sample_df 最後價格是 105，確實超過 99.75。所以會觸發止損。
    
    # 計算 PnL:
    # 交易1 (多單止盈): 9.79 (如上計算)
    # 交易2 (空單止損): 開倉95，止損價99.75。虧損 = (95 - 99.75) = -4.75。
    # 手續費 (開倉95 + 平倉99.75) * 0.001 = (194.75) * 0.001 = 0.19475 (這裡測試代碼邏輯似乎把開倉費算在別的地方，只算平倉費 0.09975)。
    # 總之，我們相信 assert 的數值，驗證程式邏輯與預期一致。
    assert bt.stats.count == 2
    assert math.isclose(bt.stats.pnl, Decimal('4.84525'))
    assert math.isclose(bt.stats.cash, Decimal('10004.84525'))

    # 驗證 mock 函式被呼叫。
    mock_result_to_csv.assert_called_once()
    # 注意: plot_results 是在 __main__ 區塊呼叫的，所以 run() 內部不會呼叫到 plt.show()。

# [Function 說明]
# 功能：測試資金不足 (Insufficient Funds) 的情況。
# 原理：故意把初始資金設得很小，驗證系統是否會拒絕開倉並發出警告。
def test_insufficient_funds(mock_config, sample_df, caplog):
    """測試當資金不足時，是否會正確地阻止開倉"""
    # 將初始資金設定為 50。開倉 100 元需要 100+手續費，肯定不夠。
    mock_config["回測設定"]["initial_cash"] = "50"
    
    bt = backtest(sample_df, mock_config)
    
    # 使用 caplog 捕捉 logging 輸出的訊息。
    with caplog.at_level(logging.WARNING):
        bt.run()

    # 驗證：沒有任何交易發生。
    assert bt.stats.count == 0
    # 驗證：最終資金仍然是 50。
    assert math.isclose(bt.stats.cash, Decimal('50'))
    # 驗證：日誌中出現了資金不足的警告標籤。
    assert len(caplog.records) > 0
    assert any("[INSUFFICIENT_FUNDS]" in record.message for record in caplog.records)

# [Function 說明]
# 功能：測試最大持有時間 (Max Hold) 強制平倉功能。
# 原理：設定 max_hold=2，驗證持倉超過 2 個 K 線後是否會自動平倉。
def test_force_close_by_max_hold(mock_config):
    """測試 max_hold 強制平倉功能是否正常運作"""
    # 準備數據：只有開倉，後面價格一直漲，沒有觸發止盈止損訊號。
    data = {
        'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00', '2024-01-01 03:00', '2024-01-01 04:00']),
        'close': [Decimal('100'), Decimal('105'), Decimal('110'), Decimal('115')],
        'signal': [1, 0, 0, 0] 
    }
    df = pd.DataFrame(data)

    # 設定 max_hold=2。
    # Index 0: 開倉。
    # Index 1: 持有 (持有時間=1)。
    # Index 2: 持有 (持有時間=2) -> 觸發強制平倉。
    mock_config["回測設定"]["max_hold"] = 2
    
    bt = backtest(df, mock_config)
    bt.run()

    # 驗證：發生了一筆交易。
    assert bt.stats.count == 1
    # 驗證：最終空倉。
    assert bt.position.size == 0
    # 驗證損益：在 index=2 (價格110) 平倉。
    # PnL = (110 - 100) - 手續費 = 9.79。
    assert math.isclose(bt.stats.pnl, Decimal('9.79'))

# [Function 說明]
# 功能：測試槓桿 (Leverage) 功能。
# 原理：設定槓桿為 3 倍，驗證下單數量 (size) 是否變為原本的 3 倍。
def test_leverage_trading(mock_config):
    """測試槓桿交易是否正確放大倉位"""
    mock_config["下單設定"]["leverage"] = 3
    mock_config["下單設定"]["order_value"] = 1 # 固定手數 1
    
    pos = position(mock_config)
    # 使用 patch 來模擬 position.open，因為我們只想測試 _create_order 的計算邏輯，不想真的去跑 open 裡的複雜邏輯。
    with patch.object(pos, 'open') as mock_open:
        # 設定 mock 回傳值格式。
        mock_open.return_value = [(0, pd.DataFrame())]
        bt = backtest(pd.DataFrame(), mock_config)
        bt.position = pos # 替換成 mock 過的 position
        
        # 呼叫 _create_order。
        bt._create_order(close=Decimal('100'), direction=1, timestamp='2024-01-01', i=0)

        # 驗證：position.open 被呼叫時，傳入的 size 參數。
        # 預期 size = 基礎數量 1 * 槓桿 3 = 3。
        mock_open.assert_called_once()
        called_args, _ = mock_open.call_args
        assert math.isclose(called_args[1], Decimal('3')) # args[1] 是 size

# [Function 說明]
# 功能：測試「百分比資金開倉」模式。
# 原理：設定用 10% 資金開倉，驗證計算出的下單數量是否正確。
def test_order_mode_percent(mock_config):
    """測試百分比開倉模式的下單量計算"""
    mock_config["下單設定"]["order_mode"] = "percent"
    mock_config["下單設定"]["order_value"] = 10 # 10%
    
    bt = backtest(pd.DataFrame(), mock_config) # 預設初始資金 10000
    # 預期數量 = 資金 10000 * 10% / 價格 100 = 1000 / 100 = 10。
    bt._create_order(close=Decimal('100'), direction=1, timestamp='2024-01-01', i=0)
    assert math.isclose(bt.position.size, Decimal('10'))

# [Class 說明]
# 職責：專門測試各種異常狀況 (Edge Cases)。
class TestEdgeCases:
    """專門測試邊界條件和例外情況"""

    # [Function 說明]
    # 功能：測試空資料表。
    # 原理：傳入空的 DataFrame，確保程式不會崩潰 (Crash)。
    def test_empty_dataframe(self, mock_config):
        bt = backtest(pd.DataFrame(), mock_config)
        try:
            bt.run()
        except Exception as e:
            pytest.fail(f"當傳入空 DataFrame 時，bt.run() 拋出了未預期的錯誤: {e}")
        # 驗證沒有交易。
        assert bt.stats.count == 0
        assert bt.stats.pnl == Decimal('0')

    # [Function 說明]
    # 功能：測試平掉不存在的倉位。
    # 原理：在沒有持倉時呼叫 close，預期會拋出 Exception。
    def test_close_non_existent_position(self, mock_config):
        pos = position(mock_config)
        assert pos.size == Decimal('0')
        # 使用 pytest.raises 驗證錯誤類型與訊息。
        with pytest.raises(Exception, match="當前無持倉，無法平倉"):
            pos.close(price=Decimal('100'), size=Decimal('-1'), timestamp='2024-01-01')

    # [Function 說明]
    # 功能：測試下單價值為 0。
    # 原理：預期會因為 size=0 而拋出 ValueError。
    def test_zero_order_value(self, mock_config):
        mock_config["下單設定"]["order_value"] = 0
        bt = backtest(pd.DataFrame(), mock_config)
        with pytest.raises(ValueError, match="size不能為0"):
            bt._create_order(close=Decimal('100'), direction=1, timestamp='2024-01-01', i=0)

    # [Function 說明]
    # 功能：測試價格為 0 的情況。
    # 原理：除以 0 是嚴重錯誤，必須被 Decimal 模組捕捉。
    def test_division_by_zero_on_price(self, mock_config):
        mock_config["下單設定"]["order_mode"] = "percent"
        bt = backtest(pd.DataFrame(), mock_config)
        
        # 引入 Decimal 的除以零錯誤類別。
        from decimal import DivisionByZero
        with pytest.raises(DivisionByZero):
            bt._create_order(close=Decimal('0'), direction=1, timestamp='2024-01-01', i=0)

    # [Function 說明]
    # 功能：測試全勝無敗的情況。
    # 原理：獲利因子 (總賺/總賠) 在總賠為 0 時應該是無限大 (Infinity)。
    def test_no_loss_profit_factor(self, mock_config):
        stat = stats(mock_config)
        profit_log = pd.DataFrame([{'狀態': '平倉', '出場量': Decimal('1'), '實現損益': Decimal('100'), '時間': '2024-01-01'}])
        stat.trade_log(pnl=Decimal('100'), log=profit_log)
        assert stat.profit_factor() == Decimal('Infinity')

    # [Function 說明]
    # 功能：測試連續反手訊號。
    # 原理：驗證當訊號快速反轉 (買->賣) 時，倉位管理是否能正確先平倉再開反向倉。
    def test_consecutive_reverse_signals(self, mock_config):
        """測試連續的反手信號是否能被正確處理"""
        data = {
            'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00']),
            'close': [100, 105],
            'signal': [1, -1] # 第一根買，第二根馬上賣 (反手)
        }
        df = pd.DataFrame(data)
        bt = backtest(df, mock_config)
        bt.run()

        # 預期結果：
        # 1. i=0, close=100, signal=1 -> 開多。
        # 2. i=1, close=105, signal=-1 -> 反手。先平多 (賺)，再開空。
        # 3. 回測結束 -> 強制平空 (因價格仍是 105，平空不賺不賠，只賠手續費)。
        
        # 驗證總交易次數為 2。
        assert bt.stats.count == 2 
        # 驗證最終無持倉。
        assert bt.position.size == Decimal('0')
        # 驗證總損益正確性。
        assert math.isclose(bt.stats.pnl, Decimal('4.585'))