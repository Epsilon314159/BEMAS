from agents.BEMAS.ddq_network import DQN 
from agents.replay_buffer import ReplayBuffer
import numpy as np
import config
from envs.scenarios.battery_endless import Scenario

FLAGS = config.flags.FLAGS
rb_capacity = FLAGS.rb_capacity
mrb_capacity = FLAGS.mrb_capacity
minibatch_size = FLAGS.minibatch_size
target_update = FLAGS.target_update

discount_factor = FLAGS.gamma

if "battery" in FLAGS.scenario:
    sup_len = 3
elif "endless" in FLAGS.scenario:
    sup_len = 3
else:
    sup_len = 0


class Agent(object):
    def __init__(self, obs_space, act_space, sess, n_agents, name):
        self.act_space = act_space
        self.n_agents = n_agents
        self.scenario = Scenario()
        self.scores_list = []
        self.BEMAS = DQN(sess, obs_space, sup_len, act_space, n_agents, name)

        self.action_rb = ReplayBuffer(capacity=rb_capacity)
       
        self.train_cnt = 0
        self.mission_train_cnt = 0
        self.sns_q = None
        self.lamda = 0.6

        self._bayes = BayesianTemporalTracker(
            n_agents=n_agents,
            n_actions=act_space,
            prior_strength=2.0,     
            memory_decay=0.95       
        )

    def reset(self):
        self.sns_q = None

    def act_multi(self, obs, random):        
        if self.sns_q is None:
            q_values = self.BEMAS.get_aq_values([obs])[0]
        else:
            q_values = self.sns_q

        r_action = np.random.randint(self.act_space, size=(len(obs)))
        action_n = ((random+1)%2)*(q_values.argmax(axis=1)) + (random)*r_action
        best_action= q_values.argmax(axis=1)
        
        return action_n, best_action
    
    
    def incentivize_multi(self, info):
        state, action, reward, next_state, done = info
        done = done.all()
            
        [ls_q, lns_q] = self.BEMAS.get_aq_values([state, next_state])
        s_q = ls_q[range(self.n_agents), action]
        ns_q = discount_factor*lns_q.max(axis=1)*(not done) + reward

        td = ns_q - s_q    

        if done:
            self.sns_q = None

        return td
    
    def add_to_memory(self, exp):
        self.action_rb.add_to_memory(exp)

    def sync_target(self):
        self.BEMAS.training_target_qnet()

    def train(self, use_rx):
        data = self.action_rb.sample_from_memory(minibatch_size)

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        base_reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])

        not_done = (done+1)%2

        if use_rx:
            rx_inc = np.asarray([x[5] for x in data])
            reward = base_reward + rx_inc
        else:
            reward = base_reward

        td_error,_ = self.BEMAS.training_a_qnet(state, action, reward, not_done, next_state)

        self.train_cnt += 1
        
        if self.train_cnt % (target_update) == 0:
            self.BEMAS.training_target_qnet()
            self.BEMAS.training_peer_qnet()

        return td_error
    
    
    def get_scores(self, evaluations):
        scores = []

        for i in range(self.n_agents):
            if evaluations[i] != 0:
                score = (1/evaluations[i]) 
            else:
                evaluations[i] = np.power(10, 8)
                score = (1/evaluations[i]) 
            scores.append(score)

        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            scores = [(x - min_score) / (max_score - min_score) for x in scores]
        else:
            scores = [0 for _ in scores]
        
        self.scores_list.append(scores)
        
        if len(self.scores_list) < 3:
            return np.array(scores)
        else: 
            scores = (1 - self.lamda) * np.array(scores) + self.lamda * np.array(self.scores_list[-2])
            return np.array(scores)


    def get_action_probs(self, obs_batch):
        arr = np.asarray(obs_batch)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        probs = self.BEMAS.get_action_probs(arr)
        if probs.ndim == 3 and probs.shape[0] == 1:
            probs = probs[0]
        return probs

    def update_bayesian_beliefs(self, policies, update_strengths=None):
        """
        policies: np.ndarray [n_agents, n_actions] = current π(a|o)
        Returns: np.ndarray [n_agents] = per-agent surprise (KL) from t-1 -> t.
        """
        return self._bayes.update(policies, update_strengths)


    def get_bayesian_temporal_penalties(self):
        """Return last computed per-agent surprise (higher => bigger penalty)."""
        return self._bayes.get_last_surprise()


class BayesianTemporalTracker:

    def __init__(self, n_agents, n_actions, prior_strength=2.0, memory_decay=0.95):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.prior_strength = float(prior_strength)
        self.memory_decay = float(memory_decay)
        self.alpha = np.ones((n_agents, n_actions), dtype=np.float32) * self.prior_strength
        self._last_surprise = np.zeros((n_agents,), dtype=np.float32)

    def reset(self):
        self.alpha[...] = self.prior_strength
        self._last_surprise[...] = 0.0

    @staticmethod
    def _kl_categorical(p, q, eps=1e-8):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        q /= q.sum(axis=1, keepdims=True)
        return np.sum(p * (np.log(p) - np.log(q)), axis=1)

    def update(self, policies, update_strengths=None):
        prior = self.alpha.copy()

        self.alpha *= self.memory_decay

        if update_strengths is None:
            w = np.ones((policies.shape[0], 1), dtype=np.float32)
        else:
            w = np.asarray(update_strengths, dtype=np.float32).reshape(-1, 1)
            w = np.maximum(w, 0.0)  
        self.alpha += w * policies

        prior_mean = prior / prior.sum(axis=1, keepdims=True)
        post_mean  = self.alpha / self.alpha.sum(axis=1, keepdims=True)
        self._last_surprise = self._kl_categorical(prior_mean, post_mean)
        return self._last_surprise

    def get_last_surprise(self):
        return self._last_surprise.copy()
