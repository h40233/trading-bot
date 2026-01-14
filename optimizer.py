# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是參數優化器模組 (Optimizer)。
# 它的功能是封裝執行策略參數優化的完整流程，包含：
# 1. 產生參數組合 (Grid Search)
# 2. 使用多核心並行運算執行回測
# 3. 處理與視覺化優化結果
# -----------------------------------------------------------------------------------------
import pandas as pd
import plotly.express as px
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from backtest import backtest

# --- 並行優化執行函式 (必須定義在最上層，才能被多進程序列化) ---
def run_optimization_task(args):
    """
    為單一參數組合執行回測的背景任務。
    """
    param_combination, selected_params, StrategyClass, raw_df, split_idx, config = args
    
    try:
        # 準備策略參數
        init_params = {name: val for name, val in zip(selected_params, param_combination)}
        strategy_instance = StrategyClass(**init_params)
        
        # 計算訊號
        temp_df = raw_df.copy()
        df_with_signal = strategy_instance.generate_signal(temp_df)
        
        # 資料分割
        df_train = df_with_signal.iloc[:split_idx]
        df_test = df_with_signal.iloc[split_idx:]
        
        # 執行回測 (in-sample)
        bt_train = backtest(df_train, config)
        bt_train.run()
        
        # 執行回測 (out-of-sample)
        bt_test = backtest(df_test, config)
        bt_test.run()
        
        # 準備結果
        result = {
            **init_params,
            "IS_Sharpe": bt_train.stats.sharpe(),
            "OS_Sharpe": bt_test.stats.sharpe(),
            "IS_PnL": bt_train.stats.pnl,
            "OS_PnL": bt_test.stats.pnl,
            "IS_Trades": bt_train.stats.count,
            "OS_Trades": bt_test.stats.count,
        }
        return result
    except Exception as e:
        print(f"優化任務失敗: 參數={param_combination}, 錯誤={e}")
        return None

class Optimizer:
    """
    參數優化器，負責執行完整的網格搜索與結果分析。
    """
    def __init__(self, config, strategy_class, raw_df):
        self.config = config
        self.StrategyClass = strategy_class
        self.raw_df = raw_df

    def run(self, selected_params, param_settings, split_ratio, progress_callback=None):
        """
        執行網格搜索優化。
        
        :param selected_params: list, 使用者選擇要優化的參數名稱列表
        :param param_settings: dict, 包含每個參數的 (start, end, step)
        :param split_ratio: float, 訓練集資料佔比
        :param progress_callback: function, 用於更新 Streamlit UI 的回呼函式
        :return: pd.DataFrame, 包含所有參數組合與其表現的結果
        """
        
        # 動態生成參數範圍
        def make_range(start, end, step):
            vals = []
            curr = start
            while curr <= end:
                is_int = step % 1 == 0 and isinstance(curr, float) and curr.is_integer()
                vals.append(int(curr) if is_int else curr)
                curr += step
            return vals

        param_ranges = [make_range(*param_settings[p]) for p in selected_params]
        param_combinations = list(itertools.product(*param_ranges))
        total_combs = len(param_combinations)
        split_idx = int(len(self.raw_df) * split_ratio)
        
        tasks = [(comb, selected_params, self.StrategyClass, self.raw_df, split_idx, self.config) for comb in param_combinations]
        results = []
        
        if progress_callback:
            progress_callback(0, f"偵測到 {multiprocessing.cpu_count()} 個 CPU 核心。準備執行 {total_combs} 組參數回測...")

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_params = {executor.submit(run_optimization_task, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(future_to_params)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    print(f'參數組合 {future_to_params[future]} 產生錯誤: {exc}')
                
                if progress_callback:
                    progress = (i + 1) / total_combs
                    progress_callback(progress, f"總體進度: {i + 1}/{total_combs} ({progress:.0%})")
        
        if not results:
            return pd.DataFrame()

        results.sort(key=lambda r: r[selected_params[0]])
        return pd.DataFrame(results)

    def plot_1d_results(self, results_df, param_name):
        """為一維優化結果生成線圖"""
        fig = px.line(results_df, x=param_name, y=["IS_Sharpe", "OS_Sharpe"], title=f"{param_name} vs Sharpe Ratio")
        fig.update_layout(xaxis_title=param_name, yaxis_title="Sharpe Ratio")
        return fig

    def plot_2d_heatmap(self, results_df, params, metric="OS_Sharpe", title_prefix=""):
        """為二維優化結果生成熱力圖"""
        x_axis, y_axis = params[0], params[1]
        title = f"{title_prefix} | {x_axis} vs {y_axis}"
        fig = px.density_heatmap(results_df, x=x_axis, y=y_axis, z=metric, text_auto=".2f", color_continuous_scale="RdBu_r", title=title)
        return fig

    def plot_parallel_coords(self, results_df, params):
        """為多維優化結果生成平行座標圖"""
        # 為了視覺化效果，將夏普比率做一點縮放，讓它在圖上更明顯
        results_df['IS_Sharpe_scaled'] = results_df['IS_Sharpe']
        results_df['OS_Sharpe_scaled'] = results_df['OS_Sharpe']
        
        fig = px.parallel_coordinates(
            results_df,
            dimensions=params + ['IS_Sharpe_scaled', 'OS_Sharpe_scaled'],
            color="OS_Sharpe_scaled",
            color_continuous_scale=px.colors.sequential.Viridis,
            title="多參數與夏普比率關係圖"
        )
        return fig
