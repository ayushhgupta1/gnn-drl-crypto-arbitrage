import argparse
import time
import torch

from trainer import Trainer
from data_feed import DataFeed
from env import ArbitrageEnv
from model import GNNArbitrageAgent


def run_demo(mode="mock"):
    # Initialize the Multi-Exchange environment and 9-Node agent
    data_feed = DataFeed(mode=mode)
    env = ArbitrageEnv(data_feed=data_feed)
    agent = GNNArbitrageAgent(num_nodes=9, node_features=5, action_dim=19)

    try:

        agent.load_state_dict(torch.load("gnn_spatial_model_15000.pth"))
        print("Loaded trained Spatial GNN model weights\n")
    except FileNotFoundError:
        print("Warning: No trained spatial model found. Using initialized weights.\n")

    print(f"Starting Multi-Exchange Spatial Routing ({mode.upper()} Mode)...")
    state = env.reset()

    if mode == "live":
        print("Waiting 3 seconds for 9-node WebSocket streams to synchronize...")
        time.sleep(3)

    for step in range(200):
        # 1. Direct PyTorch forward pass (pure exploitation, Epsilon = 0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_tensor)
            action = q_values.argmax().item()

        # 2. Decode the complex routing action
        action_info = env.action_map[action]
        action_str = action_info["name"]

        # 3. Execute the step in the environment
        next_state, reward, done = env.step(action)


        print(f"Step {step:3d} | {action_str:<45} | Global Portfolio: ${env.portfolio_value_usd:.2f}")

        if mode == "live":
            time.sleep(0.5)

        state = next_state
        if done:
            break

    print("\n" + "=" * 45)
    print("--- FINAL DISTRIBUTED INVENTORY REPORT ---")
    print("=" * 45)

    for ex in env.exchanges:
        print(f"[{ex} VAULT]")
        print(f"  USD: ${env.balances[ex]['USD']:.2f}")
        print(f"  BTC:  {env.balances[ex]['BTC']:.6f}")
        print(f"  ETH:  {env.balances[ex]['ETH']:.6f}")
        print(f"  BNB:  {env.balances[ex]['BNB']:.6f}\n")

    print("-" * 45)
    print(f"TOTAL GLOBAL PORTFOLIO NAV: ${env.portfolio_value_usd:.2f}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "demo"], default="train")
    parser.add_argument("--live", action="store_true", help="Use live Binance WebSockets + Spatial Deviations")
    args = parser.parse_args()

    # Determine the data feed mode for the demo
    data_mode = "live" if args.live else "mock"

    if args.mode == "train":
        print("Initializing Spatial DQN-GNN Training Routine...")
        trainer = Trainer()
        try:
            # Train the massive new action space
            trainer.train(episodes=15000)
        except KeyboardInterrupt:
            print("\nTraining manually interrupted by user!")
        finally:
            print("Saving model weights before exiting...")
            torch.save(trainer.policy_net.state_dict(), "gnn_spatial_model.pth")
            print("Saved as gnn_spatial_model.pth")
    else:
        run_demo(mode=data_mode)