import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, initial_std=0.0):
        """
        input_dim  : dimension d'entrée (nombre_de_frames * dimension de l'état)
        hidden_dim : dimension de la couche cachée
        output_dim : dimension de la sortie (2: accélération et angle de braquage)
        initial_std: valeur initiale de l'écart-type pour l'exploration
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # La sortie représente la moyenne des actions
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        # Paramètre de variance log, initialisé avec initial_std
        self.action_log_std = nn.Parameter(torch.ones(output_dim) * initial_std)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.action_mean(x)
        std = self.action_log_std.exp().expand_as(mean)
        return mean, std

class PolicyGradientAgent:
    def __init__(self, state_dim, frame_stack=4, hidden_dim=128, lr=1e-3, gamma=0.99, initial_std=0.0):
        """
        state_dim  : dimension de l'état initial (par exemple 4)
        frame_stack: nombre de frames accumulées pour constituer l'entrée de l'agent
        hidden_dim : taille des couches cachées du réseau
        lr         : taux d'apprentissage
        gamma      : facteur d'actualisation
        initial_std: valeur initiale de l'écart-type pour la distribution d'exploration
        """
        self.state_dim = state_dim
        self.frame_stack = frame_stack
        self.input_dim = state_dim * frame_stack
        self.gamma = gamma

        self.policy_net = PolicyNetwork(self.input_dim, hidden_dim, output_dim=2, initial_std=initial_std)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Buffer pour stocker les frames actuelles
        self.frames = deque(maxlen=frame_stack)

        # Buffers pour la politique gradient
        self.log_probs = []
        self.rewards = []

    def reset_frames(self):
        self.frames.clear()

    def push_frame(self, state):
        """
        Ajoute un nouvel état (frame) au buffer.
        Si le buffer n'est pas encore rempli, on le complète en dupliquant le state
        """
        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(state)
        else:
            self.frames.append(state)

    def get_state_stack(self):
        """
        Retourne une concaténation (flatten) des frames du buffer sous forme de tensor.
        """
        state_stack = np.concatenate(self.frames)
        return torch.FloatTensor(state_stack)

    def select_action(self, state):
        """
        Sélectionne une action en se basant sur l'état courant.
        L'action est échantillonnée d'une loi normale définie par le réseau.
        """
        self.push_frame(state)
        state_stack = self.get_state_stack().unsqueeze(0)  # ajout dimension batch

        mean, std = self.policy_net(state_stack)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # Sauvegarde le log de probabilité pour le calcul de la loss
        self.log_probs.append(dist.log_prob(action).sum(dim=-1))
        
        # Convertir l'angle de braquage de radians à degrés si besoin
        action_np = action.detach().cpu().numpy()[0]
        acceleration = action_np[0]
        steering_angle = np.degrees(action_np[1])
        return (acceleration, steering_angle)

    def finish_episode(self):
        """
        Effectue la mise à jour du réseau avec la politique gradient REINFORCE.
        """
        R = 0
        returns = []
        # Calcul des retours cumulés
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        # Normalisation des retours
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            loss -= log_prob * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset des buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.reset_frames()

# Exemple d'utilisation
if __name__ == "__main__":
    state_dim = 4
    agent = PolicyGradientAgent(state_dim=state_dim, frame_stack=4, initial_std=1.0)

    dummy_state = np.array([0.0, 5.0, 0.0, 0.0])
    num_steps = 5
    for step in range(num_steps):
        action = agent.select_action(dummy_state)
        print(f"Étape {step} | Action choisie: {action}")
        agent.rewards.append(1.0)

    agent.finish_episode()