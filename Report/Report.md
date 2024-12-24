# Reinforcement Learning Nanodegree - Reacher Unity Environment Report

## Pretrained execution

Below is the youtube video of running pretrained model.
<a href="https://www.youtube.com/watch?v=fT6UDkvKAaI">
<img src=""/>
</a>

- You can open file `Run_Pretrained.ipynb` to run it.
- pretrained model is located at [`Report/actor.pth and Report/critic.pth`](https://github.com/vyredo/nanodegree_RL_course1/blob/main/p1_navigation/runs/Banana_Linux.pt)

From the video we can see that the score for each agent are from 28 to 35

## Learning Algorithm

The agent is implemented using:

1. **DDPG Agent**:

   - The Deep Deterministic Policy Gradient (DDPG) algorithm leverages a deterministic policy for continuous action spaces.
     It uses an Actor-Critic architecture where:
     - The Actor generates the optimal action for a given state.
     - The Critic evaluates the Q-value of the action-state pair.

2. **ReplayBuffer**:

   - The Replay Buffer stores past experiences as tuples of (state, action, reward, next_state) to break temporal correlations in the data.

3. **OU Noise**:
   - It generates temporally correlated noise to encourage exploration while maintaining smooth changes in actions

## Log of Rewards

For the full log, check [this link](https://github.com/vyredo/udacity_RL_Reacher/blob/main/Report/training_logs.txt)

The whole training is done for **202 episodes**.

## Plot of Rewards

This are the plot for 202 episodes, the score from episode 186 and onwards have score that is above 30.

- Episode: 186 average last 100 Score: 31.76, done in 316.93 seconds
- Episode: 187 average last 100 Score: 31.1, done in 319.19 seconds
- Episode: 188 average last 100 Score: 31.45, done in 321.0 seconds
- Episode: 189 average last 100 Score: 32.21, done in 323.7 seconds
- Episode: 190 average last 100 Score: 33.79, done in 330.32 seconds
- Episode: 191 average last 100 Score: 32.56, done in 335.97 seconds
- Episode: 192 average last 100 Score: 32.64, done in 341.94 seconds
- Episode: 193 average last 100 Score: 33.28, done in 333.12 seconds
- Episode: 194 average last 100 Score: 32.19, done in 314.59 seconds
- Episode: 195 average last 100 Score: 31.89, done in 318.66 seconds
- Episode: 196 average last 100 Score: 33.97, done in 322.65 seconds
- Episode: 197 average last 100 Score: 33.46, done in 326.14 seconds
- Episode: 198 average last 100 Score: 31.19, done in 329.9 seconds
- Episode: 199 average last 100 Score: 32.21, done in 337.84 seconds
- Episode: 200 average last 100 Score: 33.25, done in 339.39 seconds
- Episode: 201 average last 100 Score: 32.82, done in 342.36 seconds
- Episode: 202 average last 100 Score: 32.56, done in 346.29 seconds

<img src="https://github.com/vyredo/udacity_RL_Reacher/blob/main/Report/training_progress.png" alt="Reward Plot" />
