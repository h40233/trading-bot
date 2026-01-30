# =========================================================================================
# 檔案名稱：optimizer.py
# 檔案說明：
#   本檔案實作了「參數優化器 (Optimizer)」。
#   主要功能是透過「網格搜索 (Grid Search)」演算法，窮舉所有可能的參數組合，
#   並利用 CPU 多核心平行運算加速回測過程，最終產出參數與績效的熱力圖。
#
# 核心職責：
#   1. 參數生成：利用 Cartesian Product (笛卡兒積) 產生參數組合列表。
#   2. 平行運算：使用 ProcessPoolExecutor 分派回測任務至多個 CPU 核心。
#   3. 樣本內外檢測：將數據分割為訓練集 (In-Sample) 與測試集 (Out-of-Sample) 以驗證穩健性。
#   4. 視覺化：生成熱力圖與平行座標圖，協助交易員尋找「參數高原」。
# =========================================================================================

# [模組引用解析]
# 引入 Pandas 處理優化結果的表格數據。
import pandas as pd
# 引入 Plotly 繪圖庫，用於生成互動式的 2D/3D 圖表。
import plotly.express as px
# 引入 itertools 用於高效產生參數排列組合 (product)。
import itertools
# 引入 multiprocessing 模組，用於偵測 CPU 核心數。
import multiprocessing
# 引入並行運算執行器，這是 Python 3 推薦的平行處理介面。
from concurrent.futures import ProcessPoolExecutor, as_completed

# 引入回測引擎，因為優化器本質上就是不斷地跑回測。
from backtest import backtest

# [函式說明]
# 功能：單一參數組合的回測任務。
# 注意：此函式必須定義在 Class 外部 (Module Level)，否則無法被 pickle 序列化，導致多進程失敗。 # 註1
def run_optimization_task(args):
    """
    為單一參數組合執行回測的背景任務。
    """
    # 解包傳入的參數 tuple
    param_combination, selected_params, StrategyClass, raw_df, split_idx, config = args
    
    try:
        # [變數追蹤] 初始化策略
        # 將參數名稱與數值配對 (zip)，動態傳入策略建構子。
        init_params = {name: val for name, val in zip(selected_params, param_combination)}
        strategy_instance = StrategyClass(**init_params)
        
        # [核心邏輯] 計算策略訊號
        # 必須使用 copy()，避免多個進程修改到同一份原始資料。
        temp_df = raw_df.copy()
        df_with_signal = strategy_instance.generate_signal(temp_df)
        
        # [核心邏輯] 樣本內外分割 (Train/Test Split)
        # 前段數據用於「訓練」(尋找最佳參數)，後段數據用於「測試」(驗證是否過度擬合)。
        df_train = df_with_signal.iloc[:split_idx]
        df_test = df_with_signal.iloc[split_idx:]
        
        # 執行回測 (In-Sample / 訓練集)
        bt_train = backtest(df_train, config)
        bt_train.run()
        
        # 執行回測 (Out-of-Sample / 測試集)
        bt_test = backtest(df_test, config)
        bt_test.run()
        
        # 整理回傳結果，包含參數設定與兩個階段的夏普比率。
        result = {
            **init_params, # 展開參數字典
            "IS_Sharpe": bt_train.stats.sharpe(), # 訓練集績效
            "OS_Sharpe": bt_test.stats.sharpe(),  # 測試集績效 (真實考驗)
            "IS_PnL": bt_train.stats.pnl,
            "OS_PnL": bt_test.stats.pnl,
            "IS_Trades": bt_train.stats.count,
            "OS_Trades": bt_test.stats.count,
        }
        return result
    except Exception as e:
        # 捕捉並印出錯誤，避免單一任務失敗導致整個優化崩潰。
        print(f"優化任務失敗: 參數={param_combination}, 錯誤={e}")
        return None

# =========================================================================================
# 類別：Optimizer
# 職責：優化流程管理器
# 說明：
#   負責準備資料、生成參數網格，並管理 ProcessPoolExecutor 的生命週期。
# =========================================================================================
class Optimizer:
    """
    參數優化器，負責執行完整的網格搜索與結果分析。
    """
    # [方法：初始化]
    # 接收設定檔、策略類別與原始 K 線資料。
    def __init__(self, config, strategy_class, raw_df):
        self.config = config
        self.StrategyClass = strategy_class
        self.raw_df = raw_df

    # [方法：執行優化]
    # 輸入：
    #   - selected_params: 欲優化的參數名稱列表。
    #   - param_settings: 參數範圍設定 (start, end, step)。
    #   - split_ratio: 訓練集佔比 (如 0.7 代表 70% 資料用於訓練)。
    #   - progress_callback: UI 回調函式，用於更新進度條。
    def run(self, selected_params, param_settings, split_ratio, progress_callback=None):
        """
        執行網格搜索優化。
        """
        
        # [內部函式] 生成參數範圍列表
        # 處理浮點數精度問題，並支援整數/浮點數步長。
        def make_range(start, end, step):
            vals = []
            curr = start
            while curr <= end:
                # 判斷是否應該存為整數 (例如 RSI 週期必須是 int)
                is_int = step % 1 == 0 and isinstance(curr, float) and curr.is_integer()
                vals.append(int(curr) if is_int else curr)
                curr += step
            return vals

        # 針對每個參數生成其數值列表。
        param_ranges = [make_range(*param_settings[p]) for p in selected_params]
        
        # [核心算法] 笛卡兒積 (Cartesian Product)
        # 使用 itertools.product 生成所有可能的參數排列組合。
        # 例如: RSI=[14, 20], MA=[50, 100] -> [(14,50), (14,100), (20,50), (20,100)]
        param_combinations = list(itertools.product(*param_ranges))
        total_combs = len(param_combinations)
        split_idx = int(len(self.raw_df) * split_ratio)
        
        # 準備任務列表，將所有需要的物件打包成 tuple。
        tasks = [(comb, selected_params, self.StrategyClass, self.raw_df, split_idx, self.config) for comb in param_combinations]
        results = []
        
        if progress_callback:
            progress_callback(0, f"偵測到 {multiprocessing.cpu_count()} 個 CPU 核心。準備執行 {total_combs} 組參數回測...")

        # [控制流解析] 平行運算環境
        # 使用 Context Manager (with) 確保執行完畢後資源釋放。
        # max_workers 預設為 CPU 核心數，讓運算效能最大化。
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # 提交所有任務並建立 Future 物件映射。
            future_to_params = {executor.submit(run_optimization_task, task): task for task in tasks}
            
            # 使用 as_completed 迭代已完成的任務 (誰先做完誰先回傳)。
            for i, future in enumerate(as_completed(future_to_params)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    print(f'參數組合 {future_to_params[future]} 產生錯誤: {exc}')
                
                # 更新 UI 進度
                if progress_callback:
                    progress = (i + 1) / total_combs
                    progress_callback(progress, f"總體進度: {i + 1}/{total_combs} ({progress:.0%})")
        
        # 若無結果 (全失敗)，回傳空 DataFrame。
        if not results:
            return pd.DataFrame()

        # 依第一個參數排序結果，方便查看。
        results.sort(key=lambda r: r[selected_params[0]])
        return pd.DataFrame(results)

    # [方法：繪製 1D 線圖]
    # 適用於只優化一個參數時，觀察該參數與 Sharpe 的關係。
    def plot_1d_results(self, results_df, param_name):
        """為一維優化結果生成線圖"""
        fig = px.line(results_df, x=param_name, y=["IS_Sharpe", "OS_Sharpe"], title=f"{param_name} vs Sharpe Ratio")
        fig.update_layout(xaxis_title=param_name, yaxis_title="Sharpe Ratio")
        return fig

    # [方法：繪製 2D 熱力圖]
    # 適用於優化兩個參數時，尋找顏色最深 (績效最好) 的區域。
    def plot_2d_heatmap(self, results_df, params, metric="OS_Sharpe", title_prefix=""):
        """為二維優化結果生成熱力圖"""
        x_axis, y_axis = params[0], params[1]
        title = f"{title_prefix} | {x_axis} vs {y_axis}"
        # z軸為績效指標 (Metric)，顏色越深代表數值越高。
        fig = px.density_heatmap(results_df, x=x_axis, y=y_axis, z=metric, text_auto=".2f", color_continuous_scale="RdBu_r", title=title)
        return fig

    # [方法：繪製平行座標圖]
    # 適用於優化三個以上參數時，觀察多維度參數間的關聯。
    def plot_parallel_coords(self, results_df, params):
        """為多維優化結果生成平行座標圖"""
        # 為了視覺化效果，複製 Sharpe 欄位並重新命名，避免影響原始數據。
        results_df['IS_Sharpe_scaled'] = results_df['IS_Sharpe']
        results_df['OS_Sharpe_scaled'] = results_df['OS_Sharpe']
        
        fig = px.parallel_coordinates(
            results_df,
            dimensions=params + ['IS_Sharpe_scaled', 'OS_Sharpe_scaled'],
            color="OS_Sharpe_scaled", # 線條顏色根據 OS Sharpe 決定
            color_continuous_scale=px.colors.sequential.Viridis,
            title="多參數與夏普比率關係圖"
        )
        return fig

# ====== 備註區 ======
# 註1: Multiprocessing 的序列化限制
#      在 Windows/macOS 上使用 spawn/fork 模式時，傳遞給子進程的函式必須是可被 pickle 的 (Top-level function)。
#      若將 run_optimization_task 寫在 class 內部 (instance method)，會因為包含 self 指標而導致序列化失敗。