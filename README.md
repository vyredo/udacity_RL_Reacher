# Banana Navigation - Reinforcement Learning Project

This repository contains the solution to the Udacity Nanodegree project for the Reacher Navigation environment. The agent is trained using a DDPGAgent (Dueling DQN) with Replay Buffer, The example provide is for linux OS, you will need to include another file for another OS, do include the 20 agents version.

---

## Report

Check this [link to view Report section](https://github.com/vyredo/udacity_RL_Reacher/blob/main/Report/Report.md)

## Getting Started

Follow the steps below to run the project and train the agent:

1. clone this repository with: `git clone`
2. Open the Jupyter notebook: `jupyter notebook Main.ipynb`

3. Configure your Python environment:  
   Update the notebook with the correct paths to your Python binaries and library locations. for example Replace `[USERNAME]` with your system username in the snippet below:

```python
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/home/[USERNAME]/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/home/[USERNAME]/mambaforge/envs/py310/lib/python3.10/site-packages"
```

3. Run all the cells in the notebook to start training the agent.

---

### Project Details

#### Key Features

- Algorithm: DDPG Agent with Replay Buffer for efficient learning.
- Pretrained Model: The trained model is saved at:

```bash
Report/actor.pth
Report/critic.pth
```

- Logs: Training logs are available at:

```bash
Report/training_logs.txt
```

---

### Training Progress

Below are the training results from the episode 195 to 202. The agent achive average 30 scores in these trainings :

```mathematica
Episode: 195 average last 100 Score: 31.89, done in 318.66 seconds
Episode: 196 average last 100 Score: 33.97, done in 322.65 seconds
Episode: 197 average last 100 Score: 33.46, done in 326.14 seconds
Episode: 198 average last 100 Score: 31.19, done in 329.9 seconds
Episode: 199 average last 100 Score: 32.21, done in 337.84 seconds
Episode: 200 average last 100 Score: 33.25, done in 339.39 seconds
Episode: 201 average last 100 Score: 32.82, done in 342.36 seconds
Episode: 202 average last 100 Score: 32.56, done in 346.29 seconds
```

---

### Code Overview

Here is an overview of the main components:

- **Actor-Critic-Network**:
  The Actor-Critic Network is a key component of the agent's design. It consists of two networks:

  - Actor Network: Responsible for selecting actions based on the current state.
  - Critic Network: Evaluates the quality of the state-action pair.

  Implementation of the ActorCritic can be found in:

  ```bash
  ActorCritic.py
  ```

  Implementation of the DDPGAgent can be found in:

  ```bash
  Agent.py
  ```

- **Replay Memory**:
  Code for Replay Buffer is in:

  ````bash
  ReplayBuffer.py
  ```
  ````

- **Prioritized Experience Replay**:
  I tried to use PER but the agent does not learn when PER is active and each episode takes much longer time in my system.
  The current report does not use PER, only Replay Buffer
  Code for PER is in:

  ```bash
  PrioritizeReplay.py
  ```

- **Ornstein-Uhlenbeck Noise (OU Noise)**:
  To encourage exploration in continuous action environment, the agent uses OUNoise
  Code for Replay Buffer is in:

  ```bash
  OUNoise.py
  ```

### Run Pretrained Model

1. open with Jupyter Notebook

```
Run_Pretrained.ipynb
```

2. Replace the file `Reacher.x86_64` according to your OS. The example provided is for Linux

3. Run all
