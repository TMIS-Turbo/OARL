import numpy as np
import pandas as pd
import random as rn
import argparse
import os
import gym
import matplotlib.pyplot as plt
import Environment.environment
import oarl
import torch


# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='highway-v0', help='CartPole environment')
parser.add_argument('--algo', type=str, default='oarl', help='select an algorithm among sac, asac')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--episodes', type=int, default=400, help='training episode number')
parser.add_argument('--max_step', type=int, default=200, help='max episode step')
parser.add_argument('--state_dim', type=int, default=16, help='state dimension')
parser.add_argument('--action_dim', type=int, default=1, help='action dimension')
parser.add_argument('--action_numb', type=int, default=3, help='action number')
parser.add_argument('--mode', type=str, default='train', help='train')
parser.add_argument('--save_dir_model', type=str, default='model/', help='the path to save models')
parser.add_argument('--save_dir_data', type=str, default='result/', help='the path to save data')
parser.add_argument('--save_dir_train_data', type=str, default='train/', help='the path to save training data')
args = parser.parse_args()

# Set environment
env = gym.make(args.env)

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
torch.manual_seed(args.seed)


def train():
    env.start(gui=False)
    model_l = oarl.Agent(args.state_dim, args.action_dim, args.action_numb)

    if not os.path.exists(args.save_dir_model):
        os.mkdir(args.save_dir_model)
    model_l.train()
    print("The model is training")

    score = 0.0
    total_reward = []
    episode = []
    print_interval = 10
    train_interval = 2
    interaction_times = 0

    v = []
    v_epi = []
    v_epi_mean = []

    ax = []
    ax_epi = []
    ax_epi_mean = []

    ay = []
    ay_epi = []
    ay_epi_mean = []

    cn = 0.0
    cn_epi = []

    for n_epi in range(args.episodes):
        s = env.reset()
        done = False
        step_number = 0
        while not (done or step_number == args.max_step):
            s = np.array(s, dtype=float)

            a_l = model_l.select_action_single(s, args.mode)
            s_prime, r, done, _, _, _, _ = env.step(a_l)

            model_l.replay_buffer.add(s, a_l, r, s_prime, done)

            s = s_prime
            step_number += 1
            interaction_times += 1
            score += r

            if args.mode == "train" and n_epi > print_interval and interaction_times % train_interval == 0:
                model_l.train_model()

            v.append(s[0]*35)
            v_epi.append(s[0]*35)
            ax.append(s[-2]*10)
            ax_epi.append(s[-2]*10)
            ay_ = (s[0]*35)*(s[-1]*10*3.14/180)
            ay.append(ay_)
            ay_epi.append(ay_)
            d_f = 100 * s[2]
            d_b = 100 * s[4]
            if d_f < 3 or d_b < 2.5:
                cn += 1

        if n_epi % print_interval == 0 and n_epi != 0:
            print("episode :{}, avg score_v : {:.1f}, interaction_times:{}".format(n_epi, score/print_interval, interaction_times))

            episode.append(n_epi)
            total_reward.append(score / print_interval)
            cn_epi.append(cn)

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            ax_mean = np.mean(ax_epi)
            ax_epi_mean.append(ax_mean)

            ay_mean = np.mean(ay_epi)
            ay_epi_mean.append(ay_mean)

            score = 0.0

            v_epi = []
            ax_epi = []
            ay_epi = []
            cn = 0.0

        if args.mode == "train" and (n_epi+1) % 100 == 0:
            model_l.save_model(n_epi+1, args.save_dir_model)

    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["ax_epi_mean"] = ax_epi_mean
    df["ay_epi_mean"] = ay_epi_mean
    df["cn_epi"] = cn_epi

    df_ = pd.DataFrame([])
    df_["v"] = v
    df_["ax"] = ax
    df_["ay"] = ay

    if not os.path.exists(args.save_dir_data):
        os.mkdir(args.save_dir_data)
    train_data_path = os.path.join(args.save_dir_data, args.save_dir_train_data)
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)

    df.to_csv('./' + train_data_path + '/train_rac.csv', index=0)
    df_.to_csv('./' + train_data_path + '/train_rac_.csv', index=0)

    plt.plot(episode, total_reward)
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.show()

    env.close()


if __name__ == "__main__":
    train()