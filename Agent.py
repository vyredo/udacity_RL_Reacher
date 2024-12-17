
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ActorCritic import Actor, Critic
from OUNoise import OUNoise
from PrioritizeReplay import PrioritizedReplayMemory
from ReplayBuffer import ReplayBuffer
import random
from torch.cuda.amp import GradScaler

# Code Below mainly taken from Ddpg-pendulum
# https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py


class DDPGAgent:
    def __init__(self,
                 # GLOBAL CONFIGURATION
                 DEVICE,
                 LR_ACTOR,
                 LR_CRITIC,
                 WEIGHT_DECAY,
                 BUFFER_SIZE,
                 BATCH_SIZE,
                 TAU,
                 GAMMA,
                 USE_PER, use_mixed_precision,
                 # state, action
                 state_size, action_size,
                 random_seed):
        self.TAU = TAU
        self.DEVICE = DEVICE
        self.LR_ACTOR = LR_ACTOR
        self.LR_CRITIC = LR_CRITIC
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.use_PER = USE_PER
        self.use_mixed_precision = use_mixed_precision

        self.seed = random.seed(random_seed)
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = Critic(
            state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic(
            state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # mixed precision if your GPU support it
        if self.use_mixed_precision:
            self.scaler = GradScaler()

        # Replay memory
        if self.use_PER:
            self.memory = PrioritizedReplayMemory(
                action_size,
                self.BUFFER_SIZE, self.BATCH_SIZE, device=self.DEVICE,  seed=0,)
        else:
            self.memory = ReplayBuffer(
                action_size, self.BUFFER_SIZE, self.BATCH_SIZE, 0,
                device=self.DEVICE
            )

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, current_step):
        if self.use_PER:
            self.memory.add(state, action, reward, next_state, done, priority=1.0)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn when there is enough samples
        if len(self.memory) > self.BATCH_SIZE:
            if self.use_PER:
                # learn every x steps to speed up training
                if current_step % 3 == 0:
                    experiences, weights, indices = self.memory.sample(current_step)
                    td_errors = self.learn(experiences, self.GAMMA, weights)
                    self.memory.update_priorities(indices, torch.abs(td_errors))
            else:  # learn every 10 steps to speed up training
                if current_step % 10 == 0:
                    for _ in range(2):  # Perform 2 updates per trigger
                        experiences = self.memory.sample()
                        self.learn(experiences, self.GAMMA)

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def learn(self, experiences, gamma, weights=None):
        states, actions, rewards, next_states, dones = experiences

        # in my test, it will cause the training to be slower for the initial episode
        if self.use_mixed_precision:
            # Forward pass with autocast for mixed precision
            with torch.cuda.amp.autocast():
                actions_next = self.actor_target(next_states)
                Q_targets_next = self.critic_target(next_states, actions_next)
                Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

                # Get expected Q-values from local critic
                Q_expected = self.critic_local(states, actions)

                # Critic loss (with PER weights if available)
                if weights is not None:
                    critic_loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
                else:
                    critic_loss = F.mse_loss(Q_expected, Q_targets)

                # Actor loss
                actions_pred = self.actor_local(states)
                actor_loss = -self.critic_local(states, actions_pred).mean()

            # Minimize the loss Backward pass with scaled gradients
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)

            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)

            # Update the scale for next iteration
            self.scaler.update()

        # Float 32 precision
        if not self.use_mixed_precision:
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            with torch.no_grad():
                Q_targets_next = self.critic_target(next_states, actions_next)

            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.critic_local(states, actions)

            # Compute critic loss (scaled by importance weights if using PER)
            if weights is not None:
                critic_loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
            else:
                critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

        # Compute TD errors for PER (Q_targets - Q_expected)
        if weights is not None:
            td_errors = (Q_targets - Q_expected).detach()
            return td_errors.squeeze()
