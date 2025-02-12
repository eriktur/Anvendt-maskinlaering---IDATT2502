import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# Definer DQN-nettverket
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size) # Output-noder er lik antall handlinger

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Aktiveringsfunksjonen er ReLU
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Funksjon for å velge handling basert på epsilon-greedy policy
def select_action(state, epsilon, action_size, policy_net):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # Utforsk (tilfeldig handling)
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state)
        return torch.argmax(q_values).item()  # Utnytt (velg handling med høyest anslått verdi)


# Funksjon for å lagre erfaringer i replay buffer
def store_experience(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))  # Lagrer en erfaring i replay buffer


# Funksjon for å generere en tilfeldig seed
def generate_seed():
    return random.randint(0, 100)


# Funksjon for å trene DQN-agenten basert på erfaringer fra replay buffer
def train_agent(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory) < batch_size:
        return  # Ikke nok erfaringer i replay buffer for minibatch

    # Hent et tilfeldig minibatch fra replay buffer
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(np.array(actions))
    rewards = torch.FloatTensor(np.array(rewards))
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(np.array(dones))

    # Beregn Q-verdiene for de nåværende tilstandene og handlingene
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Beregn mål-Q-verdiene for neste tilstand
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))  # Bellman-ligningen

    # Beregn tap (loss) og oppdater policy-nettverket
    loss = nn.MSELoss()(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Hyperparametere
gamma = 0.99  # Diskonteringsfaktor for fremtidige belønninger
epsilon = 1.0  # Utforskingsrate (epsilon-greedy)
epsilon_min = 0.01  # Minimum epsilon-verdi
epsilon_decay = 0.995  # Hvor raskt epsilon reduseres over tid
learning_rate = 0.001  # Læringsrate for optimizer
batch_size = 64  # Størrelse på minibatch
memory_size = 100000  # Størrelse på replay buffer
max_episodes = 1000  # Antall episoder for trening
max_steps = 650  # Maksimalt antall steg per episode

# Initialiser miljøet uten visualisering
env = gym.make("LunarLander-v3")  # Ingen render_mode for de første episodene
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialiser DQN-agenten
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Replay buffer for å lagre erfaringer
memory = deque(maxlen=memory_size)

# Treningssløyfen
for episode in range(max_episodes):
    seed = generate_seed()  # Generer seed for episoden
    if episode >= 700:
        env.close()  # Lukk tidligere miljø før du lager nytt
        env = gym.make("LunarLander-v3", render_mode="human")

    state, _ = env.reset(seed=seed)  # Tilbakestill miljøet med seed
    total_reward = 0

    for step in range(max_steps):
        if episode >= 700:
            env.render()

        # Velg handling basert på epsilon-greedy policy
        action = select_action(state, epsilon, action_size, policy_net)

        # Utfør handlingen i miljøet
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Lagre erfaringen i replay buffer
        store_experience(memory, state, action, reward, next_state, done)

        # Tren agenten med et minibatch fra replay buffer
        train_agent(memory, batch_size, policy_net, target_net, optimizer, gamma)

        state = next_state  # Oppdater tilstanden til neste tilstand

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, w/ seed: {seed}")
            break

    # Reduser epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Oppdater target-nettverket hver 10. episode
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
