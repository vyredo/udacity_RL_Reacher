import torch
import torch.nn as nn


# Responsible for generating action
# Does not process action during training
# output is mapped to action_size directly in layer 3
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, layer1_size=256, layer2_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # First layer: Learn patterns from raw inputs
        self.layer1 = nn.Linear(state_size, layer1_size)
        # Second layer: Combine patterns from Layer 1
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        # Final layer: Map patterns to action
        self.layer3 = nn.Linear(layer2_size, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        output_layer1 = self.relu(self.layer1(state))
        output_layer2 = self.relu(self.layer2(output_layer1))
        return self.tanh(self.layer3(output_layer2))


# Responsible to evaluate state-action pairs
# output a single value, which is quality of state-action pair
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, layer1_size=256, layer2_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Try learn relevant features
        self.layer1 = nn.Linear(state_size, layer1_size)
        # Second layer: Combine state and action
        self.layer2 = nn.Linear(layer1_size + action_size, layer2_size)
        # Map features to 1 value which is the quality of state-action
        self.layer3 = nn.Linear(layer2_size, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        # get only positive values ( strong signal)
        xs = self.relu(self.layer1(state))
        # relate action with the strong state
        x = torch.cat((xs, action), dim=1)
        # network will learn what's the relationship of action and state
        x = self.relu(self.layer2(x))

        return self.layer3(x)
