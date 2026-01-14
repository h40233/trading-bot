import streamlit as st
import json
import pandas as pd
import os
import glob
import inspect
from decimal import Decimal

# 引入你的專案模組
from util import load_config, to_timestamp, data_to_csv, load_strategy
from backtest import backtest
from data_loader import fetch_and_process_data
from optimizer import Optimizer
# 引入策略註冊表
from strategies import STRATEGY_REGISTRY

# --- 頁面基礎設定 ---
st.set_page_config(page_title="量化交易控制台", layout="wide", page_icon="📈")
st.title("📈 程式夥伴 - 量化交易控制台")

# --- 狀態管理 (Session State) ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- Helper: 讀取 config ---
def get_current_config():
    try:
        return load_config()
    except:
        return {}

# --- 側邊欄：系統設定 ---
st.sidebar.header("⚙️ 參數設定")
try:
    config = load_config()
except:
    st.sidebar.error("找不到 config.json")
    config = {}

with st.sidebar.form("config_form"):
    # 1. 基本設定
    with st.expander("📝 基本環境設定 (Basic)", expanded=False):
        col_b1, col_b2 = st.columns(2)
        base_conf = config.get("基本設定", {})
        symbol = col_b1.text_input("交易對", base_conf.get("symbol", "BTCUSDT"))
        timeframe = col_b2.selectbox("時間週期", ["1m", "5m", "15m", "1h", "4h", "1d"], 
                                     index=["1m", "5m", "15m", "1h", "4h", "1d"].index(base_conf.get("timeframe", "5m")))
        
        # 動態讀取所有已註冊的策略
        strategy_options = list(STRATEGY_REGISTRY.keys())
        current_strat = base_conf.get("strategy", "EMA_RSI")
        strat_index = strategy_options.index(current_strat) if current_strat in strategy_options else 0
        strategy_name = st.selectbox("選擇策略", strategy_options, index=strat_index)
        
        col_b3, col_b4 = st.columns(2)
        testnet = col_b3.checkbox("使用測試網 (Testnet)", value=base_conf.get("testnet", True))
        use_mark = col_b4.checkbox("使用標記價格 K線", value=base_conf.get("use_mark_price_kline", False))
        
        col_b5, col_b6 = st.columns(2)
        max_hold = col_b5.number_input("最大持倉 K 棒數 (0為不限)", value=int(base_conf.get("max_hold", 0) or 0))
        fetch_limit = col_b6.number_input("單次抓取 K 線數量", value=int(base_conf.get("fetch_limit", 1000)))
        
        sleep_time = st.number_input("API 冷卻秒數", value=float(base_conf.get("sleep_time", 0.5)))

    # 2. 下單設定
    with st.expander("💰 下單資金管理 (Order)", expanded=False):
        order_conf = config.get("下單設定", {})
        col_o1, col_o2 = st.columns(2)
        order_mode = col_o1.selectbox("下單模式", ["percent", "fixed", "price"], index=["percent", "fixed", "price"].index(order_conf.get("order_mode", "percent")))
        order_value = col_o2.number_input("下單數值", value=float(order_conf.get("order_value", 10)))
        leverage = st.number_input("槓桿倍數", value=int(order_conf.get("leverage", 1)))
        
        col_o3, col_o4 = st.columns(2)
        pyramiding = col_o3.checkbox("允許加倉 (Pyramiding)", value=order_conf.get("pyramiding", False))
        reverse = col_o4.checkbox("允許反手 (Reverse)", value=order_conf.get("reverse", False))

    # 3. 止盈止損設定
    with st.expander("🛡️ 止盈止損設定 (TP/SL)", expanded=False):
        tpsl_conf = config.get("止盈止損設定", {})
        col_t1, col_t2 = st.columns(2)
        tp_percent = col_t1.checkbox("TP %模式", value=tpsl_conf.get("tp_of_percent", True))
        tp_value = col_t2.number_input("TP 數值", value=float(tpsl_conf.get("tp_value", 0.0)))
        col_s1, col_s2 = st.columns(2)
        sl_percent = col_s1.checkbox("SL %模式", value=tpsl_conf.get("sl_of_percent", True))
        sl_value = col_s2.number_input("SL 數值", value=float(tpsl_conf.get("sl_value", 0.0)))

    # 4. 回測設定
    with st.expander("⏳ 回測環境設定 (Backtest)", expanded=False):
        bt_conf = config.get("回測設定", {})
        start_time_str = st.text_input("開始時間", bt_conf.get("start_time", "2023-01-01 00:00"))
        end_time_str = st.text_input("結束時間", bt_conf.get("end_time", "2023-12-31 00:00"))
        initial_cash = st.number_input("初始資金", value=float(bt_conf.get("initial_cash", 10000)))
        fee_rate = st.number_input("手續費率", value=float(bt_conf.get("fee_rate", 0.0004)), format="%.5f")
        slippage = st.number_input("滑價率", value=float(bt_conf.get("slippage", 0.0005)), format="%.5f")

    submitted = st.form_submit_button("💾 儲存並更新設定")
    
    if submitted:
        config["基本設定"].update({"symbol": symbol, "timeframe": timeframe, "strategy": strategy_name, "testnet": testnet, "use_mark_price_kline": use_mark, "max_hold": max_hold if max_hold > 0 else None, "fetch_limit": fetch_limit, "sleep_time": sleep_time})
        config["下單設定"].update({"order_mode": order_mode, "order_value": order_value, "leverage": leverage, "pyramiding": pyramiding, "reverse": reverse})
        config["止盈止損設定"].update({"tp_of_percent": tp_percent, "tp_value": tp_value, "sl_of_percent": sl_percent, "sl_value": sl_value})
        config["回測設定"].update({"start_time": start_time_str, "end_time": end_time_str, "initial_cash": initial_cash, "fee_rate": fee_rate, "slippage": slippage})
        
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        st.success("設定已儲存！")
        st.cache_data.clear()

# --- 主分頁 ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 單次回測", "🧪 參數優化實驗室", "🤖 實盤監控", "📂 檔案管理"])

# ==========================================
# 分頁 1: 單次回測 (Backtest)
# ==========================================
with tab1:
    st.subheader("歷史回測模擬")
    st.info(f"當前目標：{symbol} | 策略：{strategy_name} | 週期：{timeframe}")
    
    if st.button("🚀 開始回測", type="primary", use_container_width=True):
        status_box = st.empty()
        bar = st.progress(0)
        try:
            status_box.text("正在更新資料與計算策略...")
            bar.progress(10)
            df_signal = fetch_and_process_data(custom_config=config)
            
            if df_signal is not None:
                status_box.text("正在執行回測模擬...")
                bt = backtest(df_signal, config)
                
                def update_progress(p):
                    bar.progress(p, text=f"回測進度: {int(p*100)}%")
                bt.run(progress_callback=update_progress)
                
                bar.progress(1.0, text="回測完成！")
                status_box.success("回測完成！")
                
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 最終權益", f"{bt.stats.cash:,.2f} U", delta=f"{bt.stats.pnl:,.2f} U")
                col2.metric("🎯 勝率", f"{bt.stats.winrate():.2f} %")
                col3.metric("📉 最大回撤", f"{bt.stats.max_drawdown:,.2f} U")
                col4.metric("📊 夏普比率", f"{bt.stats.sharpe():.2f}")
                
                st.subheader("資金曲線")
                equity = bt.stats.get_equity_curve()
                if equity is not None:
                    st.line_chart(equity.set_index("時間")['資金曲線'], color="#00FF00", use_container_width=True)

                    st.subheader("水下圖 (Drawdown)")
                    st.area_chart(equity.set_index("時間")['回撤'], color="#FF0000", use_container_width=True)

                    st.subheader("週期性報酬")
                    period = st.selectbox("選擇週期", ["月", "季", "年"], index=0)
                    period_map = {"月": "M", "季": "Q", "年": "A"}
                    periodic_returns = bt.stats.get_periodic_returns(period=period_map[period])
                    if periodic_returns is not None:
                        st.bar_chart(periodic_returns.set_index("時間")['損益'], use_container_width=True)

                with st.expander("查看詳細交易日誌"):
                    st.dataframe(bt.stats.log, use_container_width=True)
            else:
                status_box.error("資料獲取失敗")
        except Exception as e:
            st.error(f"錯誤: {e}")

# ==========================================
# 分頁 2: 參數優化實驗室 (Optimization)
# ==========================================
with tab2:
    st.header("🧪 參數優化與穩健性分析")
    st.markdown("此功能自動偵測策略參數，並使用網格搜索 (Grid Search) 尋找參數高原。")
    
    if strategy_name in STRATEGY_REGISTRY:
        StrategyClass = STRATEGY_REGISTRY[strategy_name]
        
        sig = inspect.signature(StrategyClass.__init__)
        all_params = [
            p.name for p in sig.parameters.values() 
            if p.name not in ['self', 'name', 'args', 'kwargs'] 
            and p.default != inspect._empty
            and isinstance(p.default, (int, float))
        ]
        
        with st.container(border=True):
            st.subheader("1. 設定優化範圍")
            
            if not all_params:
                st.warning("此策略沒有可供優化的數值參數。")
                selected_params = []
            else:
                st.info(f"偵測到可優化參數：`{all_params}`")
                selected_params = st.multiselect(
                    "請選擇要優化的參數 (可選多個)", 
                    all_params, 
                    default=all_params[:2] if len(all_params) >= 2 else all_params
                )
            
            param_settings = {}
            if selected_params:
                cols = st.columns(len(selected_params) + 1)
                for i, param_name in enumerate(selected_params):
                    with cols[i]:
                        st.markdown(f"**{param_name}**")
                        default_val = sig.parameters[param_name].default
                        is_int = isinstance(default_val, int)
                        step_val = 1.0 if is_int else 0.1
                        
                        p_start = st.number_input(f"{param_name} 開始", value=float(default_val), key=f"p_start_{param_name}")
                        p_end = st.number_input(f"{param_name} 結束", value=float(default_val * 2), key=f"p_end_{param_name}")
                        p_step = st.number_input(f"{param_name} 間隔", value=step_val, min_value=0.0001, format="%.4f", key=f"p_step_{param_name}")
                        param_settings[param_name] = (p_start, p_end, p_step)

                with cols[-1]:
                    st.markdown("**資料分割**")
                    split_ratio = st.slider("訓練集佔比 (In-Sample %)", 0.1, 0.9, 0.7, 0.05, key="split_ratio")

        if st.button("🧪 開始網格搜索", type="primary", use_container_width=True):
            if not selected_params:
                st.error("請至少選擇一個要優化的參數！")
            else:
                status_header = st.empty()
                progress_bar = st.progress(0)
                
                status_header.text("正在獲取並清洗原始資料...")
                full_df = fetch_and_process_data(custom_config=config)
                
                if full_df is not None:
                    base_columns = ['open_time', 'open', 'high', 'low', 'close', 'close_time', 'symbol']
                    raw_df = full_df[base_columns].copy()
                    
                    opt = Optimizer(config, StrategyClass, raw_df)
                    
                    def update_ui_progress(p, text):
                        progress_bar.progress(p, text=text)

                    res_df = opt.run(selected_params, param_settings, split_ratio, progress_callback=update_ui_progress)
                    
                    status_header.success(f"✅ 已完成全部參數測試")
                    
                    if not res_df.empty:
                        st.divider()
                        st.subheader("📊 優化結果視覺化")

                        num_dims = len(selected_params)
                        
                        if num_dims == 1:
                            st.markdown("#### 一維參數掃描結果")
                            p_name = selected_params[0]
                            fig = opt.plot_1d_results(res_df, p_name)
                            st.plotly_chart(fig, use_container_width=True)

                        elif num_dims == 2:
                            st.markdown("#### 二維參數熱力圖")
                            v1, v2 = st.columns(2)
                            with v1:
                                st.subheader("🔥 訓練集 (In-Sample)")
                                fig1 = opt.plot_2d_heatmap(res_df, selected_params, metric="IS_Sharpe", title_prefix="Sharpe Ratio (Train)")
                                st.plotly_chart(fig1, use_container_width=True)
                            with v2:
                                st.subheader("❄️ 測試集 (Out-Sample)")
                                fig2 = opt.plot_2d_heatmap(res_df, selected_params, metric="OS_Sharpe", title_prefix="Sharpe Ratio (Test)")
                                st.plotly_chart(fig2, use_container_width=True)

                        else: 
                            st.markdown("#### 多維平行座標圖")
                            st.info("下圖中，每一條線代表一組參數組合。您可以拖動座標軸來篩選範圍，觀察在高 Sharpe 值時，參數大致落在哪個區間。")
                            fig = opt.plot_parallel_coords(res_df, selected_params)
                            st.plotly_chart(fig, use_container_width=True)

                        st.subheader("詳細數據")
                        st.dataframe(res_df, use_container_width=True)
                else:
                    status_header.error("資料獲取失敗，無法進行優化。")
    else:
        st.error(f"找不到策略 {strategy_name}")

# ==========================================
# 分頁 3: 實盤監控
# ==========================================
with tab3:
    st.subheader("實盤運行控制台")
    if st.session_state.is_running:
        st.success("🟢 策略執行中 (Running)")
    else:
        st.warning("🔴 策略已停止 (Stopped)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ 啟動實盤策略", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
    with col_btn2:
        if st.button("🛑 緊急停止 / Stop All", type="primary", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()

    st.write("---")
    st.markdown("### 📋 即時交易日誌")
    log_files = glob.glob("result/logs/*.csv")
    if log_files:
        latest_file = max(log_files, key=os.path.getctime)
        try:
            live_df = pd.read_csv(latest_file)
            st.dataframe(live_df.tail(15).sort_index(ascending=False), use_container_width=True)
        except:
            st.write("讀取日誌失敗")
    else:
        st.info("尚無交易紀錄")

# ==========================================
# 分頁 4: 檔案管理
# ==========================================
with tab4:
    st.subheader("本地資料管理")
    c1, c2, c3 = st.columns(3)
    raw_files = glob.glob("data/raw/*.csv")
    processed_files = glob.glob("data/processed/*.csv")
    result_files = glob.glob("result/backtests/*.csv")
    
    with c1:
        st.write(f"📁 原始 K 線 ({len(raw_files)})")
        if raw_files:
            st.dataframe(pd.DataFrame([os.path.basename(f) for f in raw_files], columns=["檔名"]), hide_index=True)
    with c2:
        st.write(f"📁 策略訊號 ({len(processed_files)})")
        if processed_files:
            st.dataframe(pd.DataFrame([os.path.basename(f) for f in processed_files], columns=["檔名"]), hide_index=True)
    with c3:
        st.write(f"📁 回測結果 ({len(result_files)})")
        if result_files:
            st.dataframe(pd.DataFrame([os.path.basename(f) for f in result_files], columns=["檔名"]), hide_index=True)