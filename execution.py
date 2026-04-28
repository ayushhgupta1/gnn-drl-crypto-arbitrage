class ExecutionEngine:
    def __init__(self):
        self.assets = ["BTC", "ETH", "BNB"]
        self.exchanges = ["BINANCE", "KRAKEN", "COINBASE"]
        self.fee = 0.0010
        self.slippage = 0.0001

        # Distributed Inventory pre-funded with liquidity
        self.inventory = {}
        for ex in self.exchanges:
            self.inventory[ex] = {
                "USD": 5000.0,
                "BTC": 0.5,
                "ETH": 10.0,
                "BNB": 50.0
            }

        # Rebuild action map to decode GNN signals
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

    def execute(self, action, order_book):
        if action == 0:
            return "WAITING (Hunting for Cross-Exchange Spreads)"

        mapping = self.action_map[action]
        asset = mapping["asset"]
        ex_buy = mapping["buy_exchange"]
        ex_sell = mapping["sell_exchange"]
        buy_node = mapping["buy_node"]
        sell_node = mapping["sell_node"]

        trade_size_usd = 1000.0  # Standard routing size per signal

        # Check Local Liquidity on the specific exchanges
        if self.inventory[ex_buy]["USD"] >= trade_size_usd:
            ask_price = order_book[buy_node, 1]
            bid_price = order_book[sell_node, 0]

            crypto_bought = (trade_size_usd * (1 - self.fee - self.slippage)) / ask_price

            if self.inventory[ex_sell][asset] >= crypto_bought:
                # ATOMIC SWAP EXECUTION
                self.inventory[ex_buy]["USD"] -= trade_size_usd
                self.inventory[ex_buy][asset] += crypto_bought

                self.inventory[ex_sell][asset] -= crypto_bought
                usd_received = crypto_bought * bid_price * (1 - self.fee - self.slippage)
                self.inventory[ex_sell]["USD"] += usd_received

                profit = usd_received - trade_size_usd
                return f"EXECUTED: {mapping['name']} | Net Swap Profit: ${profit:.2f}"
            else:
                return f"ROUTING FAILED: Insufficient {asset} liquidity on {ex_sell}"
        else:
            return f"ROUTING FAILED: Insufficient USD liquidity on {ex_buy}"

    def report(self, final_order_book):
        print("\n--- DISTRIBUTED INVENTORY REPORT ---")
        total_portfolio_usd = 0.0

        for i, ex in enumerate(self.exchanges):
            print(f"\n[{ex} VAULT]")
            ex_total = self.inventory[ex]["USD"]
            print(f"USD: ${self.inventory[ex]['USD']:.2f}")

            for a_idx, asset in enumerate(self.assets):
                node_idx = a_idx * 3 + i
                asset_val = self.inventory[ex][asset] * final_order_book[node_idx, 0]
                ex_total += asset_val
                print(f"{asset}: {self.inventory[ex][asset]:.6f} (Value: ${asset_val:.2f})")

            print(f"> {ex} Subtotal: ${ex_total:.2f}")
            total_portfolio_usd += ex_total

        print("\n" + "=" * 36)
        print(f"GLOBAL PORTFOLIO NET ASSET VALUE: ${total_portfolio_usd:.2f}")
        print("=" * 36 + "\n")