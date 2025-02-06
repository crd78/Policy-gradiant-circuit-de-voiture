import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from src.env.env import RacingEnv
from src.agent.agent import PolicyGradientAgent

# Paramètres de l'entraînement
NUM_VEHICLES = 10
NUM_EPISODES = 100
MAX_STEPS = 10000  # Nombre maximum d'étapes par épisode
FRAME_STACK = 4

# Création d'un seul environnement commun
env = RacingEnv()

# On ajoute un paramètre initial_std pour forcer une variance plus élevée au démarrage
state_dim = 4  # [x, y, vitesse, angle]
agent = PolicyGradientAgent(
    state_dim=state_dim, 
    frame_stack=FRAME_STACK, 
    hidden_dim=128, 
    lr=1e-3, 
    gamma=0.99,
    initial_std=1.0  # Augmente la variance initiale pour encourager l'exploration
)

# Configuration du graphique des récompenses
plt.ion()
reward_fig = plt.figure("Reward")
all_episode_rewards = []

tolerance = 0.1  # Exemple de valeur, définie à l'extérieur de la boucle

for episode in range(NUM_EPISODES):
    # Création de 10 véhicules dans le même environnement
    vehicles = []
    init_state = env.reset()  # On récupère un état de réinitialisation unique
    for _ in range(NUM_VEHICLES):
        frames = deque(maxlen=FRAME_STACK)
        # Ajout d'un bruit aux frames initiales
        for _ in range(FRAME_STACK):
            noise = np.random.normal(0, 0.05, init_state.shape)
            frames.append(init_state + noise)
        vehicles.append({
            'frames': frames,
            'log_probs': [],
            'rewards': [],
            'done': False,
            'lap_completed': False,
            'last_angle': np.arctan2(init_state[1], init_state[0]),
            'bonus_given': False
        })

    step = 0
    lap_triggered = False    
    episode_done = False  

    # Boucle de simulation de l'épisode
    while step < MAX_STEPS and (not lap_triggered) and (not episode_done):
        for vehicle in vehicles:
            if vehicle['done']:
                continue

            # Synchroniser le buffer interne de l'agent avec celui du véhicule
            agent.reset_frames()
            for frame in vehicle['frames']:
                agent.push_frame(frame)

            # Appeler select_action (retourne (acceleration, steering_angle en degrés))
            acceleration, steering_angle_deg = agent.select_action(vehicle['frames'][-1])
            # Ajout du log_prob généré par l'agent dans le buffer du véhicule
            vehicle['log_probs'].append(agent.log_probs[-1])
            # Convertir l'angle en radians
            steering_angle = np.radians(steering_angle_deg)
            # Appliquer un clipping aux actions
            acceleration = np.clip(acceleration, -1.0, 1.0)
            steering_angle = np.clip(steering_angle, -np.radians(30), np.radians(30))
            
            # Utiliser la méthode simulate pour mettre à jour l'état du véhicule (pas env.step)
            next_state, reward, done, _ = env.simulate(vehicle['frames'][-1], (acceleration, steering_angle))
            vehicle['rewards'].append(reward)
            vehicle['frames'].append(next_state)
            vehicle['done'] = done

            # Détection d'inactivité
            speed_threshold = 0.1  
            inactivity_limit = 50  
            if next_state[2] < speed_threshold:
                vehicle['inactive_steps'] = vehicle.get('inactive_steps', 0) + 1
            else:
                vehicle['inactive_steps'] = 0
            if vehicle.get('inactive_steps', 0) >= inactivity_limit:
                vehicle['done'] = True
                vehicle['rewards'].append(-20)

            # Détection de "teleportation"
            prev_xy = vehicle.get('prev_xy', next_state[:2])
            displacement = np.linalg.norm(next_state[:2] - prev_xy)
            teleportation_threshold = 5.0  
            if displacement > teleportation_threshold:
                vehicle['done'] = True
                vehicle['rewards'].append(-10)
            vehicle['prev_xy'] = next_state[:2]

            # Rendu visuel (optionnel)
            env.render(scale_x=2, scale_y=2)

            # Vérifier le franchissement de la ligne d'arrivée
            current_angle = np.arctan2(next_state[1], next_state[0])
            tolerance = 0.2  
            if (not vehicle.get('lap_completed', False)) and (vehicle['last_angle'] > tolerance) and (abs(current_angle) < tolerance):
                vehicle['lap_completed'] = True
                lap_triggered = True
            vehicle['last_angle'] = current_angle

            if vehicle['done']:
                episode_done = True
                break

        if episode_done or any(v['done'] for v in vehicles):
            break

        step += 1

    # Bonus de fin d'épisode pour franchissement de ligne d'arrivée
    if lap_triggered:
        bonus = tolerance * avg_episode_reward
        for vehicle in vehicles:
            curr_angle = np.arctan2(vehicle['frames'][-1][1], vehicle['frames'][-1][0])
            if (not vehicle.get('bonus_given', False)) and (abs(curr_angle) < tolerance):
                vehicle['rewards'].append(20)
                vehicle['bonus_given'] = True

    # Calcul et mise à jour de la politique
    episode_rewards = [sum(v['rewards']) for v in vehicles]
    avg_episode_reward = np.mean(episode_rewards)
    all_episode_rewards.append(avg_episode_reward)

    policy_loss = 0
    for vehicle in vehicles:
        R = 0
        returns = []
        for r in vehicle['rewards'][::-1]:
            R = r + agent.gamma * R
            returns.insert(0, R)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = torch.FloatTensor(returns)
        for log_prob_val, R_val in zip(vehicle['log_probs'], returns):
            policy_loss -= log_prob_val * R_val

    agent.optimizer.zero_grad()
    policy_loss.backward()
    agent.optimizer.step()

    # Mise à jour du graphique des récompenses
    plt.figure("Reward")
    plt.clf()
    plt.title("Entraînement de l'agent (Policy Gradient)")
    plt.xlabel("Episode")
    plt.ylabel("Reward moyen")
    plt.plot(all_episode_rewards, '-o')
    plt.draw()              # Forcer le redessin du graphique
    plt.pause(0.01)         # Pause légèrement plus longue pour assurer l'actualisation
    print(f"Episode {episode+1}/{NUM_EPISODES} - Reward moyen : {avg_episode_reward:.2f}")

plt.ioff()
plt.show()