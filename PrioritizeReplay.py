import torch
import random
from collections import namedtuple

# I use PER for the Course 2 for Banana Unity.
# Below is modified PER to use torch


class PrioritizedReplayMemory:

    def __init__(self, action_size, buffer_size, batch_size, seed, device='cpu', alpha=0.6):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device

        self.seed = random.seed(seed)

        self.memory = []
        self.priorities = torch.zeros(
            buffer_size, dtype=torch.float32, device=device)

        self.position = 0
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done, priority=1.0):
        """Add a new experience to memory with a given priority."""
        # Convert inputs to torch tensors
        # stays in cpu to reduce overhead moving between cpu and gpu
        # move to gpu when learn()
        state = torch.tensor(state, dtype=torch.float32, device='cpu')
        action = torch.tensor(action, dtype=torch.float32, device='cpu')
        reward = torch.tensor([reward], dtype=torch.float32, device='cpu')
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device='cpu')
        done = torch.tensor([float(done)], dtype=torch.float32, device='cpu')

        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.position] = e

        # Update priority
        self.priorities[self.position] = max(
            priority, 1e-5)  # Avoid zero priority
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, frame_idx, beta_start=0.4, beta_frames=100000):
        current_size = len(self.memory)
        if current_size < self.batch_size:
            raise ValueError(
                f"Not enough samples to draw from. Need at least {self.batch_size}, got {current_size}.")

        beta = min(1.0, beta_start + frame_idx *
                   (1.0 - beta_start) / beta_frames)

        scaled_priorities = self.priorities[:current_size] ** self.alpha
        sampling_probs = scaled_priorities / (scaled_priorities.sum() + 1e-5)

        indices = torch.multinomial(
            sampling_probs, self.batch_size, replacement=False).to(self.device)

        # Gather sampled transitions
        experiences = [self.memory[idx] for idx in indices.tolist()]

        #  move to gpu when needed (learn() is not always for every step)
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.stack([e.action for e in experiences]).to(self.device)
        rewards = torch.stack([e.reward for e in experiences]).to(self.device)
        next_states = torch.stack(
            [e.next_state for e in experiences]).to(self.device)
        dones = torch.stack([e.done for e in experiences]).to(self.device)

        # Compute importance-sampling weights
        total = current_size
        weights = (total * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()

        return (states, actions, rewards, next_states, dones), weights.unsqueeze(1), indices

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            p = priority.item() if isinstance(priority, torch.Tensor) else float(priority)
            self.priorities[idx] = max(p, 1e-5)  # Avoid zero priority

    def __len__(self):
        return len(self.memory)
