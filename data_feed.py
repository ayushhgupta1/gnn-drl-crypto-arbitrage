import numpy as np
import json
import threading
import websocket
import time
import random


class DataFeed:
    def __init__(self, mode="mock"):
        self.mode = mode
        self.assets = ["BTC", "ETH", "BNB"]
        self.exchanges = ["BINANCE", "KRAKEN", "COINBASE"]

        # 3 Assets * 3 Exchanges = 9 Nodes
        self.symbols = [f"{a}-{e}" for a in self.assets for e in self.exchanges]
        self.num_nodes = len(self.symbols)
        self.num_features = 4
        self.step_counter = 0

        # Anchor prices (Base prices for BTC, ETH, BNB)
        base_prices = [65000.0, 3500.0, 600.0]
        self.initial_prices = np.zeros(self.num_nodes)

        # Populate initial prices for all 9 nodes
        for i, asset in enumerate(self.assets):
            for j, exchange in enumerate(self.exchanges):
                idx = i * len(self.exchanges) + j
                self.initial_prices[idx] = base_prices[i]

        self.base_prices = self.initial_prices.copy()

        self.live_data = np.zeros((self.num_nodes, self.num_features))
        for i in range(self.num_nodes):
            # [Bid, Ask, Bid_Vol, Ask_Vol]
            self.live_data[i] = [self.initial_prices[i], self.initial_prices[i] + 0.1, 1.0, 1.0]

        if self.mode == "live":
            self._start_websockets()

    def reset_mock(self):
        """Called at the start of every episode to kill the price drift bug"""
        self.base_prices = self.initial_prices.copy()
        self.step_counter = 0

    def get_order_book(self):
        if self.mode == "mock":
            return self._mock_l2_data()
        else:
            return self.live_data.copy()

    def _mock_l2_data(self):
        self.step_counter += 1
        noise = np.random.normal(0, 0.0005, self.num_nodes)
        self.base_prices *= (1 + noise)

        l2_data = np.zeros((self.num_nodes, self.num_features))
        for i in range(self.num_nodes):
            spread = self.base_prices[i] * 0.0001
            bid = self.base_prices[i] - spread
            ask = self.base_prices[i] + spread
            bid_vol = np.random.uniform(0.5, 5.0)
            ask_vol = np.random.uniform(0.5, 5.0)
            l2_data[i] = [bid, ask, bid_vol, ask_vol]

        # Inject Synthetic SPATIAL Arbitrage
        # We force a massive price difference between Binance and Kraken
        cycle = self.step_counter % 40

        # BTC Binance vs Kraken
        if cycle in [0, 1, 2]:
            l2_data[0, 1] *= 0.95  # Binance BTC Ask drops
            l2_data[1, 0] *= 1.05  # Kraken BTC Bid spikes

        # ETH Coinbase vs Binance
        elif cycle in [20, 21, 22]:
            l2_data[5, 1] *= 0.95  # Coinbase ETH Ask drops
            l2_data[3, 0] *= 1.05  # Binance ETH Bid spikes

        return l2_data

    def _start_websockets(self):
        # We connect to Binance to get the heartbeat of the market
        binance_symbols = [f"{sym.lower()}usdt@bookTicker" for sym in self.assets]
        streams = "/".join(binance_symbols)
        url = f"wss://stream.binance.com:9443/ws/{streams}"

        def on_message(ws, message):
            data = json.loads(message)
            sym = data.get('s', '')

            asset_idx = -1
            if sym == "BTCUSDT":
                asset_idx = 0
            elif sym == "ETHUSDT":
                asset_idx = 1
            elif sym == "BNBUSDT":
                asset_idx = 2

            if asset_idx != -1:
                base_bid = float(data['b'])
                base_ask = float(data['a'])

                # Update Binance Node (Exact Live Data)
                binance_node_idx = asset_idx * 3
                self.live_data[binance_node_idx] = [base_bid, base_ask, float(data['B']), float(data['A'])]

                # Simulate Kraken Node (Live Binance Data + Random Micro-Deviation)
                kraken_node_idx = asset_idx * 3 + 1
                k_dev = random.uniform(-0.0005, 0.0005)
                self.live_data[kraken_node_idx] = [base_bid * (1 + k_dev), base_ask * (1 + k_dev), 1.0, 1.0]

                # Simulate Coinbase Node (Live Binance Data + Random Micro-Deviation)
                coinbase_node_idx = asset_idx * 3 + 2
                c_dev = random.uniform(-0.0005, 0.0005)
                self.live_data[coinbase_node_idx] = [base_bid * (1 + c_dev), base_ask * (1 + c_dev), 1.0, 1.0]

        def run_ws():
            while True:
                try:
                    ws = websocket.WebSocketApp(url, on_message=on_message)
                    ws.run_forever()
                except Exception:
                    time.sleep(2)  # Reconnect if connection drops

        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()