import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_numb, hidden_sizes):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.pi = nn.Linear(hidden_sizes, action_numb)

    def forward(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.pi(x)

        prob = F.softmax(x, dim=softmax_dim)
        return prob


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_numb, hidden_sizes):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, action_numb)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        q = self.fc2(x)
        return q


class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(device),
                    done=torch.Tensor(self.done_buf[idxs]).to(device))


class Agent():
    """
    An implementation of Soft Actor-Critic (SAC), Automatic entropy adjustment SAC (ASAC)
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_numb,
                 gamma=0.95,
                 hidden_sizes=128,
                 buffer_size=int(1000000),
                 batch_size=128,
                 actor_lr=1e-4,
                 qf_lr=1e-3,
                 dual_cst_lr=5e-4,
                 target_robust_error=0.0001,
                 attack_optimizing_times=5,
                 ):
        super(Agent, self).__init__()

        self.state_dim = state_dim
        self.act_dim = action_dim
        self.action_numb = action_numb
        self.gamma = gamma
        self.hidden_sizes = hidden_sizes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.qf_lr = qf_lr
        self.dual_cst_lr = dual_cst_lr
        self.attack_optimizing_times = attack_optimizing_times
        self.target_robust_error = target_robust_error

        # Main network
        self.actor = ActorNet(self.state_dim, self.action_numb, self.hidden_sizes).to(device)
        self.qf1 = CriticNet(self.state_dim, self.action_numb, self.hidden_sizes).to(device)
        self.qf2 = CriticNet(self.state_dim, self.action_numb, self.hidden_sizes).to(device)
        # Target network
        self.qf1_target = CriticNet(self.state_dim, self.action_numb, self.hidden_sizes).to(device)
        self.qf2_target = CriticNet(self.state_dim, self.action_numb, self.hidden_sizes).to(device)

        # Initialize target parameters to match main parameters
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=self.qf_lr)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(self.state_dim, self.act_dim, self.buffer_size)

        # dual variable
        self.dual_cst = torch.ones(1, requires_grad=True, device=device)
        self.dual_cst_optimizer = optim.Adam([self.dual_cst], lr=self.dual_cst_lr)

    def train_model(self):
        batch = self.replay_buffer.sample(self.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        # Prediction π(s), logπ(s), π(s'), logπ(s'), Q1(s,a), Q2(s,a)
        a, prob = self.select_action_batch(obs1)
        a_next, prob_next = self.select_action_batch(obs2)

        q1 = self.qf1(obs1).gather(1, acts.long()).squeeze(1)
        q2 = self.qf2(obs1).gather(1, acts.long()).squeeze(1)
        q1_pi_next = self.qf1_target(obs2)
        q2_pi_next = self.qf2_target(obs2)

        # Min Double-Q: min(Q1(s,π(s)), Q2(s,π(s))), min(Q1‾(s',π(s')), Q2‾(s',π(s')))
        min_q_next_pi = torch.min(q1_pi_next, q2_pi_next).to(device)
        max_js_d = self.get_optimal_perturb_Bayes(prob, prob_next, obs1, obs2)

        # Targets for Q and V regression
        v_backup = (prob_next*min_q_next_pi).sum(dim=-1) - self.dual_cst.exp()*max_js_d
        q_backup = rews + self.gamma * (1 - done) * v_backup
        q_backup.to(device)

        # Soft actor losses
        with torch.no_grad():
            q1_pi = self.qf1(obs1)
            q2_pi = self.qf2(obs1)
            min_q_pi = torch.min(q1_pi, q2_pi).to(device)
        actor_loss = (self.dual_cst.exp()*max_js_d - (prob*min_q_pi).sum(dim=-1)).mean()

        # Update actor network parameter
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft critic losses
        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())

        # Update two Q network parameter
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # constraint loss
        cst_loss = (self.dual_cst.exp()*(self.target_robust_error-max_js_d.detach())).mean()

        # Update dual constraint
        self.dual_cst_optimizer.zero_grad()
        cst_loss.backward()
        self.dual_cst_optimizer.step()

        # Polyak averaging for target parameter
        self.soft_target_update(self.qf1, self.qf1_target)
        self.soft_target_update(self.qf2, self.qf2_target)

    def get_optimal_perturb_Bayes(self, prob, prob_next, obs1, obs2):
        self.prob = prob
        self.prob_next = prob_next
        self.obs1 = obs1
        self.obs2 = obs2
        target_list = []
        target_list_grads = []
        perturb_list = []

        pbounds = {'u1': (0.8, 1.2), 'u2': (-0.05, 0.05)}
        optimizer = BayesianOptimization(f=self.js_d_loss, pbounds=pbounds, random_state=0)
        util = UtilityFunction(kind='ucb', kappa=1.0, xi=0.1)
        for i in range(self.attack_optimizing_times):
            probe_para = optimizer.suggest(util)
            target = self.js_d_loss(**probe_para)
            optimizer.register(probe_para, target.item())

            target_list.append(target.item())
            target_list_grads.append(target)
            perturb_list.append(probe_para)

        return max(target_list_grads)

    def js_d_loss(self, u1, u2):
        perturb_a, perturb_prob = self.select_action_batch(u1 * self.obs1 + u2)
        perturb_a_next, perturb_prob_next = self.select_action_batch(u1 * self.obs2 + u2)

        p_mean = (self.prob + perturb_prob) / 2
        js_d = 0.5 * torch.sum(self.prob * torch.log(self.prob / p_mean), dim=1) + 0.5 * torch.sum(
            perturb_prob * torch.log(perturb_prob / p_mean), dim=1)
        js_d = js_d.unsqueeze(1)

        p_mean_next = (self.prob_next + perturb_prob_next) / 2
        js_d_next = 0.5 * torch.sum(self.prob_next * torch.log(self.prob_next / p_mean_next), dim=1) + 0.5 * torch.sum(
            perturb_prob_next * torch.log(perturb_prob_next / p_mean_next), dim=1)
        js_d_next = js_d_next.unsqueeze(1)

        return (js_d + js_d_next).mean()

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def select_action_batch(self, state):
        state = torch.FloatTensor(state).to(device)
        prob = self.actor(state, softmax_dim=1)
        m = Categorical(prob)
        action = m.sample().reshape([-1, 1])
        return action, prob

    def select_action_single(self, state, mode='train'):
        state = torch.FloatTensor(state).to(device)
        prob = self.actor(state, softmax_dim=0)
        m = Categorical(prob)
        action = m.sample()
        a = action.detach().cpu().numpy()
        if mode == 'test':
            a = torch.argmax(prob, dim=0).item()
        return a

    def train(self, mode: bool = True) -> "OARL":
        self.actor.train(mode)
        self.qf1.train(mode)
        self.qf2.train(mode)
        return self

    def save_model(self, model_name, model_path):
        name = './' + model_path + '/policy%d' % model_name
        torch.save(self.actor, "{}.pkl".format(name))
        print("The model is saved!!!")




