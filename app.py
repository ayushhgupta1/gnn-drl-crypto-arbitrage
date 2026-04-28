import streamlit as st
import time
import threading
import torch
import pandas as pd
import plotly.graph_objects as go

from env import ArbitrageEnv
from data_feed import DataFeed
from model import GNNArbitrageAgent


@st.cache_resource
def get_global_state():
    state = {
        "portfolio_value": 0.0,
        "starting_value": None,  # FIX: We will capture the true exact value dynamically
        "portfolio_history": [],
        "balances": {},
        "prices": {},
        "history": [],
        "status": "Initializing Spatial Routing Engine..."
    }

    exchanges = ["BINANCE", "KRAKEN", "COINBASE"]
    assets = ["BTC", "ETH", "BNB"]

    for ex in exchanges:
        state["balances"][ex] = {"USD": 5000.0, "BTC": 0.5, "ETH": 10.0, "BNB": 50.0}

    for a in assets:
        for ex in exchanges:
            state["prices"][f"{a}-{ex}"] = 0.0

    return state


GLOBAL_STATE = get_global_state()


# 2. Pass the secured state dictionary directly into the background thread
def trading_thread(shared_state):
    data_feed = DataFeed(mode="live")
    env = ArbitrageEnv(data_feed=data_feed)

    agent = GNNArbitrageAgent(num_nodes=9, node_features=5, action_dim=19)

    try:
        agent.load_state_dict(torch.load("gnn_spatial_model_15000.pth"))
        shared_state["status"] = "Spatial Market Stream Connected. GNN Routing Active."
    except FileNotFoundError:
        shared_state["status"] = "Warning: No trained spatial model found. Using initialized weights."

    state = env.reset()
    time.sleep(3)

    env.current_ob = data_feed.get_order_book()
    env.portfolio_value_usd = env._calculate_total_value(env.current_ob)

    # 1. Lock in the true starting NAV
    shared_state["starting_value"] = env.portfolio_value_usd
    shared_state["portfolio_value"] = env.portfolio_value_usd

    # This destroys any "mock" prices that the UI accidentally graphed during the 3-second sleep.
    shared_state["portfolio_history"] = [env.portfolio_value_usd]

    step = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_tensor)
            action = q_values.argmax().item()

        action_str = env.action_map[action]["name"]
        next_state, reward, done = env.step(action)

        shared_state["portfolio_value"] = env.portfolio_value_usd

        shared_state["portfolio_history"].append(env.portfolio_value_usd)
        if len(shared_state["portfolio_history"]) > 100:
            shared_state["portfolio_history"].pop(0)

        for ex in env.exchanges:
            shared_state["balances"][ex]["USD"] = env.balances[ex]["USD"]
            shared_state["balances"][ex]["BTC"] = env.balances[ex]["BTC"]
            shared_state["balances"][ex]["ETH"] = env.balances[ex]["ETH"]
            shared_state["balances"][ex]["BNB"] = env.balances[ex]["BNB"]

        for a_idx, asset in enumerate(env.assets):
            for e_idx, ex in enumerate(env.exchanges):
                node_idx = a_idx * 3 + e_idx
                shared_state["prices"][f"{asset}-{ex}"] = env.current_ob[node_idx, 0]

        log_entry = {
            "Step": step,
            "Action": action_str,
            "Global NAV": f"${env.portfolio_value_usd:.2f}"
        }

        shared_state["history"].insert(0, log_entry)
        if len(shared_state["history"]) > 50:
            shared_state["history"].pop()

        state = next_state
        step += 1

        if done:
            state = env.reset()

        time.sleep(0.5)


# 3. Start the thread safely using caching
@st.cache_resource
def start_background_thread(_state_dict):
    t = threading.Thread(target=trading_thread, args=(_state_dict,), daemon=True)
    t.start()
    return t


start_background_thread(GLOBAL_STATE)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Spatial Arbitrage Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stMetric { background-color: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #c9d1d9; font-family: monospace; }
    .vault-header { color: #58a6ff; font-weight: bold; font-family: monospace; border-bottom: 1px solid #30363d; padding-bottom: 5px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("GNN MULTI-EXCHANGE ROUTING TERMINAL")
st.caption(GLOBAL_STATE["status"])
st.markdown("---")


baseline = GLOBAL_STATE["starting_value"] if GLOBAL_STATE["starting_value"] is not None else 300000.00
net_profit = GLOBAL_STATE['portfolio_value'] - baseline

st.metric("Global Net Asset Value (NAV)", f"${GLOBAL_STATE['portfolio_value']:,.2f}", f"{net_profit:,.2f} USD")
st.markdown("<br>", unsafe_allow_html=True)

# ROW 2: DISTRIBUTED INVENTORY VAULTS
exchanges = ["BINANCE", "KRAKEN", "COINBASE"]
v_cols = st.columns(3)

for i, ex in enumerate(exchanges):
    with v_cols[i]:
        st.markdown(f"<div class='vault-header'>[{ex} VAULT]</div>", unsafe_allow_html=True)
        inner_cols = st.columns(2)
        inner_cols[0].metric("USD", f"${GLOBAL_STATE['balances'][ex]['USD']:,.2f}")
        inner_cols[1].metric("BTC", f"{GLOBAL_STATE['balances'][ex]['BTC']:.4f}")
        inner_cols[0].metric("ETH", f"{GLOBAL_STATE['balances'][ex]['ETH']:.4f}")
        inner_cols[1].metric("BNB", f"{GLOBAL_STATE['balances'][ex]['BNB']:.4f}")

st.markdown("<hr style='border:1px solid #30363d'>", unsafe_allow_html=True)

# ROW 3: CHART AND LOGS
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Live Global Equity Curve")

    line_color = '#3fb950' if net_profit >= 0 else '#f85149'
    fig = go.Figure()

    history_data = GLOBAL_STATE['portfolio_history']
    if len(history_data) > 0:
        fig.add_trace(go.Scatter(
            y=history_data,
            mode='lines',
            line=dict(color=line_color, width=3),
            fill='tozeroy',
            fillcolor=f"rgba({63 if net_profit >= 0 else 248}, {185 if net_profit >= 0 else 81}, {80 if net_profit >= 0 else 73}, 0.1)"
        ))

        # Calculate dynamic zoom bounds so the chart doesn't anchor to 0
        min_y = min(history_data)
        max_y = max(history_data)
        buffer = max(5.0, (max_y - min_y) * 0.5)

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            height=300,
            xaxis_title="Execution Steps",
            yaxis_title="USD Value",
            yaxis=dict(range=[min_y - buffer, max_y + buffer]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Live Spatial Order Book (Bid Prices)")
    p_cols = st.columns(3)

    with p_cols[0]:
        st.caption("BINANCE FEED")
        st.write(f"**BTC**: ${GLOBAL_STATE['prices']['BTC-BINANCE']:,.2f}")
        st.write(f"**ETH**: ${GLOBAL_STATE['prices']['ETH-BINANCE']:,.2f}")
        st.write(f"**BNB**: ${GLOBAL_STATE['prices']['BNB-BINANCE']:,.2f}")

    with p_cols[1]:
        st.caption("KRAKEN FEED")
        st.write(f"**BTC**: ${GLOBAL_STATE['prices']['BTC-KRAKEN']:,.2f}")
        st.write(f"**ETH**: ${GLOBAL_STATE['prices']['ETH-KRAKEN']:,.2f}")
        st.write(f"**BNB**: ${GLOBAL_STATE['prices']['BNB-KRAKEN']:,.2f}")

    with p_cols[2]:
        st.caption("COINBASE FEED")
        st.write(f"**BTC**: ${GLOBAL_STATE['prices']['BTC-COINBASE']:,.2f}")
        st.write(f"**ETH**: ${GLOBAL_STATE['prices']['ETH-COINBASE']:,.2f}")
        st.write(f"**BNB**: ${GLOBAL_STATE['prices']['BNB-COINBASE']:,.2f}")

with c2:
    st.subheader("Neural Network Routing Log")
    if GLOBAL_STATE["history"]:
        df = pd.DataFrame(GLOBAL_STATE["history"])
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)
    else:
        st.info("Awaiting routing blocks...")

time.sleep(0.5)
st.rerun()