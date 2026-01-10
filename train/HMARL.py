import torch
import json
import os
import help_func as hf
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from simulations import DroneSimulation


# --- GLOBAL CONFIG & CLASSES ---
torch.autograd.set_detect_anomaly(True)

class Inter_Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Inter_Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, output_dim)
        self.log_std_layer = nn.Linear(64, output_dim)

    def custome_Sigmoid(self, upper_lim, lower_lim, x, c=1.2):
        x = ((upper_lim - lower_lim) / (1 + torch.exp(-c * x))) + lower_lim
        return x

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_layer(x)
        upper_lim = torch.full_like(mean, 80)
        lower_lim = torch.full_like(mean, 0)
        upper_lim[1::2] = 65
        mean = self.custome_Sigmoid(upper_lim, lower_lim, mean)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-7, max=2)
        std = torch.exp(log_std)
        return mean, std


class Detect_Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Detect_Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, output_dim)
        self.log_std_layer = nn.Linear(64, output_dim)

    def custome_Sigmoid(self, upper_lim, lower_lim, x, c=1.0):
        x = ((upper_lim - lower_lim) / (1 + torch.exp(-c * x))) + lower_lim
        return x

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_layer(x)
        # Define bounds
        upper_lim = torch.full_like(mean, 80)
        lower_lim = torch.full_like(mean, 0)
        pattern = torch.tensor([80, 65, 155], device=mean.device)
        upper_lim = pattern.repeat(mean.shape[-1] // 3)
        lower_lim[2::3] = 25
        mean = self.custome_Sigmoid(upper_lim, lower_lim, mean)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-7, max=2)
        std = torch.exp(log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


def flatten_list(list_scores):
    flattened_list = []
    for item in list_scores:
        if isinstance(item, tuple):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list


class PPO:
    def __init__(self, actor, critic, actor_lr=2.5e-4, critic_lr=1e-3, gamma=0.99, clip_eps=0.2):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def select_action(self, state_tensor, episode, monitor, i, j):
        mean, std = self.actor(state_tensor)
        normal_dist = dist.Normal(mean, std)
        action = normal_dist.rsample()
        action = action.view(i, j)
        action = action.detach().cpu().numpy()
        value = self.critic(state_tensor).detach().cpu().numpy()
        monitor.append({
            'episode': episode,
            'mean': mean.detach().cpu().numpy().tolist(),
            'std': std.detach().cpu().numpy().tolist(),
            'state': action.tolist()
        })
        return action, value, monitor

    def train(self, state_tensor, next_state_tensor, reward, value, next_value):
        advantage = reward + self.gamma * next_value - value
        mean, std = self.actor(state_tensor.flatten())
        dist_current = dist.Normal(mean, std)
        predicted_positions = dist_current.sample()
        old_log_probs = dist_current.log_prob(predicted_positions.flatten()).sum(dim=-1).detach()

        for _ in range(5):
            mean_new, std_new = self.actor(next_state_tensor.flatten())
            dist_new = dist.Normal(mean_new, std_new)
            new_predict_pos = dist_new.sample()
            new_log_probs = dist_new.log_prob(new_predict_pos.flatten()).sum(dim=-1).detach()
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            advantage = torch.tensor(advantage, requires_grad=True).to(device)
            policy_loss = -torch.min(policy_ratio * advantage, clipped_ratio * advantage)

            current_value_prediction = self.critic(state_tensor.flatten())
            target_value = reward + self.gamma * next_value

            target_value = torch.tensor(target_value, dtype=torch.float32).to(device).detach()

            value_loss = (current_value_prediction - target_value) ** 2

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    for critic_lr in [1e-5]:
        for inter_actor_lr in [1e-5]:
            for detect_actor_lr in [2.5e-4]:

                script_dir = os.path.dirname(os.path.abspath(__file__))
                excel_file = os.path.join(script_dir, '..', 'init/town_data.xlsx')
                town_df = pd.read_excel(os.path.abspath(excel_file))
                towns = town_df.to_dict(orient='records')
                town_edges = hf.get_town_edges(towns)

                selected_targets = [(3, 30), (10, 10), (25, 25), (50, 22), (55, 40), (30, 40), (60, 20), (65, 42)]

                excel_file = os.path.join(script_dir, '..', 'init/placements.xlsx')
                interdetec_df = pd.read_excel(os.path.abspath(excel_file), sheet_name="Placements")
                interceptors = interdetec_df["Interceptors"].apply(
                    lambda x: eval(x) if isinstance(x, str) else x).tolist()
                detectors = interdetec_df["Detectors"].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()

                if isinstance(interceptors[0], list): interceptors = interceptors[0]
                if isinstance(detectors[0], list): detectors = detectors[0]

                num_interceptors = int(interdetec_df["Number of Interceptors"].iloc[0])
                num_detectors = int(interdetec_df["Number of Detectors"].iloc[0])

                extras_detect = 2
                extras_intercept = 1
                intercept_radius = 20
                detector_range = 40
                border_y = 70
                interceptor_delay = 250
                time_step = 0.2

                inter_state = interceptors
                detectors_state = detectors

                if extras_detect:
                    num_detectors += extras_detect
                    for _ in range(extras_detect):
                        detectors_state.extend([(40, 30, 90)])
                if extras_intercept:
                    num_interceptors += extras_intercept
                    for _ in range(extras_intercept):
                        inter_state.extend([(40, 30)])

                print(inter_state)
                print(detectors_state)

                simulation = DroneSimulation(time_step, inter_state, detectors_state, detector_range, selected_targets,
                                             interceptor_delay, border_y)

                inter_actor = Inter_Actor(num_interceptors * 3 + num_detectors * 3, num_interceptors * 2).to(device)
                inter_critic = Critic(num_interceptors * 3 + num_detectors * 3).to(device)
                detect_actor = Detect_Actor(num_interceptors * 3 + num_detectors * 3, num_detectors * 3).to(device)
                detect_critic = Critic(num_interceptors * 3 + num_detectors * 3).to(device)

                simulation.level = 4
                simulation.init_level2()

                inter_agent = PPO(inter_actor, inter_critic, actor_lr=inter_actor_lr, critic_lr=critic_lr)
                detect_agent = PPO(detect_actor, detect_critic, actor_lr=detect_actor_lr, critic_lr=critic_lr)

                inter_rewards_log = []
                detect_rewards_log = []
                inter_avg_reward_log = []
                detect_avg_reward_log = []
                inter_episodes_log = []
                detect_episodes_log = []
                level_log = []
                inter_rate_log = []
                best_configs = []
                inter_monitor = []
                detect_monitor = []
                dump = []
                dump_float = 0
                how_many_100 = 0
                last_updated = 0

                inter_rewards, detect_rewards, inter_scores, detect_scores, dump_float = simulation.simulate()

                for episode in range(10000):

                    flattened_inter_scores = flatten_list(inter_scores)
                    flatten_detectors_state = flatten_list(detectors_state)
                    current_state = flattened_inter_scores + flatten_detectors_state
                    state_tensor = torch.FloatTensor(current_state).to(device)


                    inter_state, inter_value, inter_monitor = inter_agent.select_action(state_tensor.flatten(), episode,
                                                                                        inter_monitor, i=-1, j=2)
                    detect_state, detect_value, detect_monitor = detect_agent.select_action(
                        state_tensor.flatten(), episode, detect_monitor, i=-1, j=3)

                    value = inter_value
                    inter_state = [tuple(pair.tolist()) for pair in inter_state]
                    inter_state = [(x, y) for (x, y) in inter_state]

                    detect_state = [tuple(pair.tolist()) for pair in detect_state]
                    detectors_state = [(x, y, theta) for (x, y, theta) in detect_state]


                    simulation.update_interceptors(inter_state)
                    simulation.update_detectors(detect_state)

                    inter_rewards, detect_rewards, inter_scores, detect_scores, inter_rate = simulation.simulate()

                    inter_rate_log.append(inter_rate)
                    penalty = hf.compute_density_penalty(inter_state, min_dist=intercept_radius - 10,
                                                         penalty_factor=1.5)
                    inter_rewards = inter_rewards - penalty

                    # Compute penalty
                    detect_penalty = hf.compute_detect_density(detectors_state, detector_range, 30)
                    detect_rewards = detect_rewards - detect_penalty

                    total_detect_rewards = inter_rewards + detect_rewards

                    flattened_inter_scores = flatten_list(inter_scores)
                    flatten_detectors_state = flatten_list(detectors_state)
                    current_state = flattened_inter_scores + flatten_detectors_state
                    next_state_tensor = torch.FloatTensor(current_state).to(device)
                    inter_next_state_pred, next_value, dump = inter_agent.select_action(next_state_tensor.flatten(),
                                                                                        episode, dump, i=-1, j=2)
                    detect_next_state_pred, detect_next_value, dump = detect_agent.select_action(
                        next_state_tensor.flatten(), episode, dump, i=-1, j=3)

                    inter_episodes_log.append(episode)
                    detect_episodes_log.append(episode)
                    inter_rewards_log.append(inter_rewards)
                    detect_rewards_log.append(detect_rewards)
                    level_log.append(simulation.level)
                    inter_agent.train(state_tensor, next_state_tensor, inter_rewards, value, next_value)
                    detect_agent.train(state_tensor, next_state_tensor, total_detect_rewards, detect_value,
                                       detect_next_value)

                    if len(inter_rewards_log) >= 50:
                        inter_avg_reward = sum(inter_rewards_log[-50:]) / 50
                        how_many_100 = inter_rate_log.count(100.0)
                        if inter_rate == 100:
                            print(f'!!!!!   Interception 100%   !!!!!')
                            print(f'Number of 100% = {how_many_100}')
                        inter_avg_reward_log.append(inter_avg_reward)
                        print(
                            f"Episode {episode}, Reward: {inter_rewards}, Interception Rate: {inter_rate}, Avg Last 50: {inter_avg_reward} Level {simulation.level}")
                    else:
                        inter_avg_reward = sum(inter_rewards_log) / len(inter_rewards_log)
                        print(
                            f"Episode {episode}, Reward: {inter_rewards}, Interception Rate: {inter_rate}, Level {simulation.level}")
                        inter_avg_reward_log.append(inter_avg_reward)

                    if len(detect_rewards_log) >= 50:
                        detect_avg_reward = sum(detect_rewards_log[-50:]) / 50
                        detect_avg_reward_log.append(detect_avg_reward)
                        print(
                            f"Detect Episode {episode}, Detect Reward: {detect_rewards}, Avg Last 50: {detect_avg_reward}")
                    else:
                        detect_avg_reward = sum(detect_rewards_log) / len(detect_rewards_log)
                        print(f"Detect Episode {episode}, Detect Reward: {detect_rewards}")
                        detect_avg_reward_log.append(detect_avg_reward)

                    best_configs = hf.save_best_config(best_configs, inter_state, detectors_state, inter_rewards,
                                                       inter_avg_reward, simulation.level, episode=episode)

                    # Level Up Logic
                    if (((inter_avg_reward >= 1500 or how_many_100 > 50) and last_updated + 200 <= episode) or
                            (simulation.level >= 4 and (
                                    inter_avg_reward >= 1100 or how_many_100 > 10) and last_updated + 200 <= episode)):
                        hf.save_models(inter_actor, inter_critic, detect_actor, detect_critic, simulation.level)
                        hf.save_to_file(best_configs, simulation.level)
                        break

                    if (inter_avg_reward <= 1000 and how_many_100 < 10) and last_updated + 1000 <= episode:
                        how_many_100 = 0
                        simulation.leveldown()
                        simulation.levelup()
                        if simulation.init_level2():
                            last_updated = episode
                            print("Level Reset!!!!")
                            inter_actor = Inter_Actor(num_interceptors * 3 + num_detectors * 3,
                                                      num_interceptors * 2).to(device)
                            # inter_critic = Critic(num_interceptors * 3 + num_detectors * 3).to(device)
                            detect_actor = Detect_Actor(num_interceptors * 3 + num_detectors * 3, num_detectors * 3).to(
                                device)
                            # detect_critic = Critic(num_interceptors * 3 + num_detectors * 3).to(device)
                            inter_agent = PPO(inter_actor, inter_critic, actor_lr=inter_actor_lr, critic_lr=critic_lr)
                            detect_agent = PPO(detect_actor, detect_critic, actor_lr=detect_actor_lr,
                                               critic_lr=critic_lr)
                        else:
                            break

                # After training, save to JSON
                with open('inter_mean_std_log.json', 'w') as f:
                    json.dump(inter_monitor, f, indent=2, sort_keys=True)
                with open('detect_mean_std_log.json', 'w') as f:
                    json.dump(detect_monitor, f, indent=2, sort_keys=True)

                # Save final best configurations to file
                hf.save_to_file(best_configs, simulation.level)
                hf.save_models(inter_actor, inter_critic, detect_actor, detect_critic, simulation.level)

                # plot interception rewards
                plt.figure(figsize=(10, 5))
                plt.plot(inter_episodes_log, inter_avg_reward_log, label='Interception Avg Rewards',
                         color='b', linewidth=2)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards")
                plt.title(f"Inter Lr = {inter_actor_lr}, Det Lr = {detect_actor_lr}, Crit Lr = {critic_lr}")
                plt.legend()
                plt.grid(True)

                # Generate filename with learning rates
                filename = f"Level{simulation.level}_Train_intercept.png"
                plt.savefig(filename, dpi=300)
                plt.close()  # Close plot to free memory

                # plot detection rewards
                plt.figure(figsize=(10, 5))
                plt.plot(detect_episodes_log, detect_avg_reward_log, label='Detection Avg Rewards',
                         color='b', linewidth=2)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards")
                plt.title(f"Inter Lr = {inter_actor_lr}, Det Lr = {detect_actor_lr}, Crit Lr = {critic_lr}")
                plt.legend()
                plt.grid(True)

                # Generate filename with learning rates
                filename = f"Level{simulation.level}_Train_detect.png"
                plt.savefig(filename, dpi=300)
                plt.close()  # Close plot to free memory

                # Create level figure
                plt.figure(figsize=(10, 5))
                plt.plot(inter_episodes_log, level_log, label='Level', color='b', linewidth=2)
                plt.xlabel("Episodes")
                plt.ylabel("Level")
                plt.title(f"Inter Lr = {inter_actor_lr}, Det Lr = {detect_actor_lr}, Crit Lr = {critic_lr}")
                plt.legend()
                plt.grid(True)

                # Generate filename with learning rates
                filename = f"Level{simulation.level}_levelTrain.png"
                plt.savefig(filename, dpi=300)
                plt.close()  # Close plot to free memory

