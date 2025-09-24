from agents.BEMAS.agent import Agent
from agents.base_trainer import Trainer as BaseTrainer
from agents.base_trainer import stringify
from agents.simple_agent import RunningAgent as NonLearningAgent
import numpy as np
import tensorflow as tf
import config
from datetime import datetime
from envs.scenarios.battery_endless import Scenario
from envs.environment import MultiAgentEnv
np.set_printoptions(precision=3)
from envs.grid_core import World
FLAGS = config.flags.FLAGS
minibatch_size = FLAGS.minibatch_size
n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
test_interval = FLAGS.test_interval
train_interval = FLAGS.train_interval
quota = FLAGS.max_quota
import math


class Trainer(BaseTrainer):
    def __init__(self, environment, logger):
        self.env = environment
        self.logger = logger
        self.n_agents = n_predator + n_prey
        self.scenario = Scenario()
        
        self.encounter_all, self.encounter_step = self.scenario.encounters_count()[0], self.scenario.encounters_count()[1]

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

        self._agent_profile = self.env.get_agent_profile()
        agent_precedence = self.env.agent_precedence
        self.predator_singleton = Agent(act_space=self._agent_profile["predator"]["act_spc"],
                                        obs_space=self._agent_profile["predator"]["obs_dim"],
                                        sess=self.sess, n_agents=n_predator,
                                        name="predator")

        self.agents = []
        for i, atype in enumerate(agent_precedence):
            if atype == "predator":
                agent = self.predator_singleton
            else:
                agent = NonLearningAgent(self._agent_profile[atype]["act_spc"])

            self.agents.append(agent)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

        if FLAGS.load_nn:
            if FLAGS.nn_file == "":
                logger.error("No file for loading Neural Network parameter")
                exit()
            self.saver.restore(self.sess, FLAGS.nn_file)
        else:
            self.predator_singleton.sync_target()

    def density_within_observable_space(self, agent_positions, observable_range=(7, 7)):
        densities = []
        for x, y in agent_positions:
            count = 0
            x_min, x_max = x - observable_range[0] // 2, x + observable_range[0] // 2
            y_min, y_max = y - observable_range[1] // 2, y + observable_range[1] // 2
            for nx, ny in agent_positions:
                if x_min <= nx <= x_max and y_min <= ny <= y_max and (nx, ny) != (x, y):
                    count += 1
            densities.append(count)
        return densities
    
    def get_neighbors_within_observable_space(self, position, predator_positions, observable_range=(7, 7)):
        neighbors = []
        x, y = position
        x_min, x_max = x - observable_range[0] // 2, x + observable_range[0] // 2
        y_min, y_max = y - observable_range[1] // 2, y + observable_range[1] // 2
        for i, (nx, ny) in enumerate(predator_positions):
            if x_min <= nx <= x_max and y_min <= ny <= y_max and (nx, ny) != position:
                neighbors.append((i, nx, ny))
        return neighbors
    
    def get_worst_neighbors(self, predator_positions, phase_scores, observable_range=(7,7)):
        worst = {}
        for i, pos in enumerate(predator_positions):
            neighs = self.get_neighbors_within_observable_space(pos, predator_positions, observable_range)
            if not neighs:
                worst[i] = None
            else:
                j, _, _ = max(neighs, key=lambda x: phase_scores[x[0]])
                worst[i] = j
        return worst


    def get_best_neighbors(self, predator_positions, phase_scores, observable_range=(7,7)):
        worst = {}
        for i, pos in enumerate(predator_positions):
            neighs = self.get_neighbors_within_observable_space(pos, predator_positions, observable_range)
            if not neighs:
                worst[i] = None
            else:
                j, _, _ = min(neighs, key=lambda x: phase_scores[x[0]])
                worst[i] = j
        return worst


    def get_top_3_neighbors_with_highest_phase_scores(self, predator_positions, phase_scores, observable_range=(7, 7)):
        top_neighbors = {}
        for agent_idx, position in enumerate(predator_positions):
            neighbors = self.get_neighbors_within_observable_space(position, predator_positions, observable_range)
            neighbor_phase_scores = []
            for neighbor_idx, nx, ny in neighbors:
                neighbor_phase_scores.append((neighbor_idx, phase_scores[neighbor_idx]))

            neighbor_phase_scores.sort(key=lambda x: x[1], reverse=True)
            top_neighbors[agent_idx] = neighbor_phase_scores[:4]
        return top_neighbors
    

    def distance_to_closest_agent(self, predator_positions, prey_positions):
        distances = []
        all_positions = predator_positions + prey_positions
        for px, py in predator_positions:
            closest_distance = np.inf
            for ax, ay in all_positions:
                if (px, py) == (ax, ay):
                    continue
                distance = np.sqrt((px - ax)**2 + (py - ay)**2)
                if distance < closest_distance:
                    closest_distance = distance
            distances.append(closest_distance)
        return distances


    def get_incentives(self, info_n):
        inc_n = self.predator_singleton.incentivize_multi(info_n)
        inc_n = inc_n.tolist()

        for agent in self.agents[n_predator:]:
            inc_n.append(0)

        return inc_n


    def learn(self, max_global_steps, max_step_per_ep):
        epsilon = 1.0
        epsilon_dec = 1.0/(FLAGS.explore)
        epsilon_min = 0.1

        start_time = datetime.now()

        if max_global_steps % test_interval != 0:
            max_global_steps += test_interval - (max_global_steps % test_interval)

        steps_before_train = min(FLAGS.minibatch_size*4, FLAGS.rb_capacity)

        tds = []

        ep = 0
        global_step = 0
        while global_step < max_global_steps:
            ep += 1
            obs_n = self.env.reset()
            self.predator_singleton.reset()

            for step in range(max_step_per_ep):
                global_step += 1                
                rx_inc_new_final_list = []  
                inc_n = []

                predator_positions = self.env.get_predator_positions()

                act_n = self.get_actions(obs_n, epsilon)[0]
                obs_n_next, reward_n, done_n, _ = self.env.step(act_n) 

                done = done_n[:n_predator].all()
                done_n[:n_predator] = done

                predator_positions = self.env.get_predator_positions()

                transition = [obs_n[:n_predator], act_n[:n_predator], 
                    reward_n[:n_predator], obs_n_next[:n_predator], done_n[:n_predator]]
                
                inc_n = self.get_incentives(transition)
                
                phase_scores = self.predator_singleton.get_scores(evaluations=inc_n[:n_predator])

                if global_step < 500000:
                    worst_map = self.get_best_neighbors(predator_positions, phase_scores)
                    beta = 0.1
                
                else:
                    worst_map = self.get_worst_neighbors(predator_positions, phase_scores) 
                    beta = 1 - (0.9 * (max_global_steps - global_step) / (max_global_steps - 500000))
                
                obs_agents = obs_n[:n_predator]                    
                π_agent   = self.predator_singleton.get_action_probs(obs_agents)

                worst_obs = []
                for i in range(n_predator):
                    j = worst_map[i]
                    if j is None:
                        worst_obs.append(np.zeros_like(obs_agents[0]))
                    else:
                        worst_obs.append(obs_n[j])

                worst_obs = np.stack(worst_obs, axis=0)               

                π_worst = self.predator_singleton.get_action_probs(worst_obs)

                eps = 1e-8
                π_a = π_agent + eps
                π_b = π_worst  + eps
                kl_per_agent = np.sum(π_a * np.log(π_a/π_b), axis=1) 
                
                bayes_surprise = self.predator_singleton.update_bayesian_beliefs(π_agent)

                obs_agents = obs_n[:n_predator]
                act_agents = act_n[:n_predator]
                q_values_all = self.predator_singleton.BEMAS.get_aq_values(np.expand_dims(obs_agents, axis=0))  # shape: (n_predator, n_actions)
                q_values_chosen = np.array([q_values_all[0][i][act_agents[i]] for i in range(n_predator)])  # shape: (n_predator,)

                if global_step < 500000:
                    best_neighbors = self.get_worst_neighbors(predator_positions, phase_scores)
                    alpha = 0.1 + (0.9 * (500000 - global_step) / (500000))
                
                else:
                    best_neighbors = self.get_best_neighbors(predator_positions, phase_scores)
                    alpha = 0.1

                discrepancy = np.zeros(n_predator)
                for i in range(n_predator):
                    neighbor_idx = best_neighbors[i]
                    if neighbor_idx is not None:
                        neighbor_q = q_values_all[0][neighbor_idx][act_agents[neighbor_idx]]
                        discrepancy[i] = q_values_chosen[i] - neighbor_q
                    else:
                        discrepancy[i] = 0.0

                opt = np.zeros(n_predator)
                for i in range(n_predator):
                    if discrepancy[i] > 0:
                        opt[i] = alpha * 0.9 * np.log(discrepancy[i])
                    else:
                        opt[i] = 0.0

                rx_inc_new_final_list = (opt + beta * kl_per_agent - bayes_surprise).tolist()

                exp = transition + [rx_inc_new_final_list[:n_predator]]
                
                self.predator_singleton.add_to_memory(exp)


                if global_step > steps_before_train and global_step % train_interval == 0:
                    td = self.predator_singleton.train(global_step>50000)
                    tds.append(td)      

                if global_step % test_interval == 0:
                    mean_steps, mean_b_reward, mean_captured, success_rate, rem_bat = self.test(25, max_step_per_ep)
                
                    time_diff = datetime.now() - start_time
                    start_time = datetime.now()

                    est = (max_global_steps - global_step)*time_diff/test_interval 
                    etd = est + start_time
                    td = np.asarray(tds).mean(axis=1)
                    self.logger.info("%d\ttd_er\t%s" %(global_step, stringify(td[:n_predator], "\t")))
                    print(global_step, ep, "%0.2f"%(mean_steps), mean_b_reward[:n_predator], "%0.2f"%mean_b_reward[:n_predator].mean(), "%0.2f"%epsilon)
                    print("estimated time remaining %02d:%02d (%02d:%02d)"%(est.seconds//3600,(est.seconds%3600)//60,etd.hour,etd.minute))
                
                
                    self.logger.info("%d\tsteps\t%0.2f" %(global_step, mean_steps))
                    self.logger.info("%d\tb_rwd\t%s" %(global_step, stringify(mean_b_reward[:n_predator],"\t")))
                    self.logger.info("%d\tcaptr\t%s" %(global_step, stringify(mean_captured[:n_predator], "\t")))
                    self.logger.info("%d\tsuccs\t%s" %(global_step, stringify(success_rate[:n_predator], "\t")))
                    # self.logger.info("%d\tpr_ev\t%s" %(global_step, stringify(mean_peer_eval[:n_predator], "\t")))
                    self.logger.info("%d\tbttry\t%s" %(global_step, stringify(rem_bat, "\t")))
                    self.logger.info("%d\tRX\t%s" %(global_step, stringify(rx_inc_new_final_list[:n_predator], "\t")))


                    tds = []

                if done or global_step == max_global_steps: 
                    #print("donestep", step)
                    break

                obs_n = obs_n_next
                epsilon = max(epsilon_min, epsilon - epsilon_dec)

    def test(self, max_ep, max_step_per_ep, max_steps=10000):
        if max_steps < max_step_per_ep:
            max_steps = max_global_steps

        total_b_reward_per_episode = np.zeros((max_ep, self.n_agents))
        total_captured_per_episode = np.zeros((max_ep, self.n_agents))
        success_rate_per_episode = np.zeros((max_ep, self.n_agents))
        remaining_battery = np.zeros((n_predator))

        total_steps_per_episode = np.ones(max_ep)*max_step_per_ep

        global_step = 0
        ep_finished = max_ep
        for ep in range(max_ep):

            if global_step > max_steps:
                ep_finished = ep
                break

            obs_n = self.env.reset()
            self.predator_singleton.reset()

            for step in range(max_step_per_ep):
                global_step += 1

                act_n = self.get_actions(obs_n)[0]

                obs_n_next, reward_n, done_n, info_n = self.env.step(act_n)
                done = done_n[:n_predator].all()

                total_b_reward_per_episode[ep] += reward_n

                if done: 
                    break

                obs_n = obs_n_next

            if "battery" in FLAGS.scenario:
                for i in range(n_predator):
                    remaining_battery[i] += obs_n_next[i][-3]

            total_captured_per_episode[ep] = info_n
            
            success_rate_per_episode[ep, :n_predator] = 1*(total_captured_per_episode[ep, :n_predator] >= quota)

            total_steps_per_episode[ep] = step+1

        mean_b_reward = total_b_reward_per_episode[:ep_finished].mean(axis=0)
        mean_captured = total_captured_per_episode[:ep_finished].mean(axis=0)
        success_rate = success_rate_per_episode[:ep_finished].mean(axis=0)
        mean_steps = total_steps_per_episode[:ep_finished].mean()
        remaining_battery = remaining_battery/ep_finished

        return mean_steps, mean_b_reward, mean_captured, success_rate, remaining_battery
    
