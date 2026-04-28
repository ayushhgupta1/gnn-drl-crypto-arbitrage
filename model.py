import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()

    def forward(self, x, adj):
        # x: (Batch, Nodes, Features)
        # adj: (Batch, Nodes, Nodes)
        support = self.linear(x)
        out = torch.bmm(adj, support)
        return self.activation(out)


class GNNArbitrageAgent(nn.Module):
    def __init__(self, num_nodes=9, node_features=5, action_dim=19):
        super().__init__()

        self.num_nodes = num_nodes

        # Graph processing (Extracting spatial imbalances across exchanges)
        self.gcn1 = GCNLayer(node_features, 16)
        self.gcn2 = GCNLayer(16, 32)

        # Flatten graph features into linear layer
        self.fc_input_dim = num_nodes * 32

        # Dueling DQN architecture
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Fully connected adjacency matrix (all 9 nodes interact with each other)
        adj = torch.ones(batch_size, self.num_nodes, self.num_nodes).to(x.device)

        # Graph Message Passing
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)

        x = x.view(batch_size, -1)

        # Dueling DQN split
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Aggregate
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values