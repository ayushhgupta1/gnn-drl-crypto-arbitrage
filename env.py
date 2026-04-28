import numpy as np


class ArbitrageEnv:
    def __init__(self, data_feed):
        self.data_feed = data_feed
        self.n_nodes = data_feed.num_nodes  # Now 9 nodes

        # Standard crypto exchange taker fee (0.1%)
        self.taker_fee = 0.0010
        self.slippage_penalty = 0.0001

        # Spatial Arbitrage Actions: 1 WAIT + 18 EXECUTE PAIRS
        self.action_space = 19

        self.assets = ["BTC", "ETH", "BNB"]
        self.exchanges = ["BINANCE", "KRAKEN", "COINBASE"]

        # Build the Action Map for 18 possible cross-exchange routing paths
        self.action_map = {0: {"name": "WAIT (Holding Capital)"}}
        idx = 1
        for a_idx, asset in enumerate(self.assets):
            for i, ex_buy in enumerate(self.exchanges):
                for j, ex_sell in enumerate(self.exchanges):
                    if i != j:
                        buy_node = a_idx * 3 + i
                        sell_node = a_idx * 3 + j
                        self.action_map[idx] = {
                            "name": f"{asset}: Buy {ex_buy} -> Sell {ex_sell}",
                            "asset": asset,
                            "buy_exchange": ex_buy,
                            "sell_exchange": ex_sell,
                            "buy_node": buy_node,
                            "sell_node": sell_node
                        }
                        idx += 1
        self.reset()

    def reset(self):
        if self.data_feed.mode == "mock":
            self.data_feed.reset_mock()

        self.current_ob = self.data_feed.get_order_book()
        self.current_step = 0

        # Distributed Inventory Setup
        # We pre-fund every exchange with $5,000 USD and an equivalent reserve of Crypto
        self.balances = {}
        for ex in self.exchanges:
            self.balances[ex] = {
                "USD": 5000.0,
                "BTC": 0.5,
                "ETH": 10.0,
                "BNB": 50.0
            }

        self.portfolio_value_usd = self._calculate_total_value(self.current_ob)
        return self._get_state()

    def _calculate_total_value(self, ob):
        total = 0.0
        for i, ex in enumerate(self.exchanges):
            total += self.balances[ex]["USD"]
            # Value the stored crypto using the specific Bid price on that specific exchange
            total += self.balances[ex]["BTC"] * ob[0 * 3 + i, 0]
            total += self.balances[ex]["ETH"] * ob[1 * 3 + i, 0]
            total += self.balances[ex]["BNB"] * ob[2 * 3 + i, 0]
        return total

    def _get_state(self):
        state = np.zeros((self.n_nodes, 5))
        for a_idx, asset in enumerate(self.assets):
            for e_idx, ex in enumerate(self.exchanges):
                node_idx = a_idx * 3 + e_idx
                base_price = self.data_feed.base_prices[node_idx]

                # Feature 1-4: Normalized Order Book Data
                state[node_idx, 0] = self.current_ob[node_idx, 0] / base_price
                state[node_idx, 1] = self.current_ob[node_idx, 1] / base_price
                state[node_idx, 2] = self.current_ob[node_idx, 2]
                state[node_idx, 3] = self.current_ob[node_idx, 3]

                # Feature 5: Normalized Local Inventory
                # (Helps the network learn if an exchange is running out of ammo)
                if asset == "BTC":
                    initial = 0.5
                elif asset == "ETH":
                    initial = 10.0
                else:
                    initial = 50.0

                state[node_idx, 4] = self.balances[ex][asset] / initial

        return state

    def step(self, action):
        self.current_step += 1
        next_ob = self.data_feed.get_order_book()

        reward = 0.0
        trade_size_usd = 1000.0  # Fixed $1000 bet size per spatial arbitrage attempt

        if action > 0:
            mapping = self.action_map[action]
            asset = mapping["asset"]
            ex_buy = mapping["buy_exchange"]
            ex_sell = mapping["sell_exchange"]
            buy_node = mapping["buy_node"]
            sell_node = mapping["sell_node"]

            # Check Physical Constraints: Does the Buy Exchange have USD?
            if self.balances[ex_buy]["USD"] >= trade_size_usd:
                ask_price = self.current_ob[buy_node, 1]
                bid_price = self.current_ob[sell_node, 0]

                # Amount of crypto we get after paying the entry fee
                crypto_bought = (trade_size_usd * (1 - self.taker_fee - self.slippage_penalty)) / ask_price

                # Check Physical Constraints: Does the Sell Exchange have Crypto to dump?
                if self.balances[ex_sell][asset] >= crypto_bought:

                    # 1. Execute Buy Leg
                    self.balances[ex_buy]["USD"] -= trade_size_usd
                    self.balances[ex_buy][asset] += crypto_bought

                    # 2. Execute Sell Leg
                    self.balances[ex_sell][asset] -= crypto_bought
                    usd_received = crypto_bought * bid_price * (1 - self.taker_fee - self.slippage_penalty)
                    self.balances[ex_sell]["USD"] += usd_received

                    # Reward is purely the Realized PnL of the atomic swap
                    pnl = usd_received - trade_size_usd
                    reward = pnl / 10.0
                else:
                    # Attempted to route a sell without inventory
                    reward = -2.0
            else:
                # Attempted to route a buy without USD inventory
                reward = -2.0
        else:
            # Micro-penalty for waiting to encourage finding spreads
            reward = -0.01

        self.current_ob = next_ob
        self.portfolio_value_usd = self._calculate_total_value(self.current_ob)

        # Episode ends if the bot blows up the portfolio or reaches 200 steps
        done = self.portfolio_value_usd < 10000.0 or self.current_step >= 200

        return self._get_state(), reward, done