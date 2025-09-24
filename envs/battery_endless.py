import numpy as np
from envs.grid_core import World
from envs.scenarios.pursuit_battery import Scenario as BaseScenario
from envs.scenarios.pursuit_battery import Prey as BasePrey
from envs.scenarios.pursuit_base import Predator as Agent
import config
import time

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
map_size = FLAGS.map_size

max_step_per_ep = FLAGS.max_step_per_ep
history_len = FLAGS.history_len

power_dec = 1
power_threshold = 0

class Predator(Agent):
    def __init__(self, power_threshold=0, power_dec=1):
        super(Predator, self).__init__(obs_range=3)
        self.power_threshold = power_threshold
        self.power_dec = power_dec
        self.power = 100
        self.step = 0.0
        self.gathered = 0
        self.obs_dim = (((self.obs_range*2 + 1)**2)*3 + 3)*history_len
        self.involved = 0
        self.alpha = 0.6  
        self.gathered_moving_avg = 0.0  
        self.start_time = time.time() 

    @property
    def p_pos(self):
        return self._x, self._y

    def get_obs(self):
        return np.array(self._obs).flatten()

    def update_obs(self, obs):
        self.step += 1.0
        self.grid = obs
        obs = np.append(obs.bin_encode().flatten(),\
            [self.power/100.0, self.step/100.0, self.gathered/10.0])
        self._obs.append(obs)

    def base_reward(self, capture, involved, is_terminal):
        self.gathered += capture
        self.involved += involved

        reward = 0

        if self.action.u != 2:
            self.power -= self.power_dec

        if FLAGS.rwd_form == "siglin":
            if is_terminal: # siglin
                if self.power > 0:
                    reward += 100./(1. + np.exp(-(self.power-30)/10.0))
                reward += self.gathered*12

        if FLAGS.rwd_form == "picsq":
            if is_terminal: #picsq
                if self.power > 0:
                    reward += 100*(self.power > 50)
                reward +=5*self.gathered**2

        if FLAGS.rwd_form == "sigsig":
            if is_terminal: #sigsig
                if self.power > 0:
                    reward += 100./(1. + np.exp(-(self.power-70)/10.0))
                reward += 150./(1. + np.exp(-(self.gathered-5)))

        return reward

    def is_done(self):
        return False

    def reset(self):
        self.power = 100
        self.gathered = 0
        self.involved = 0
        self.step = 0.0

class Prey(BasePrey):
    def __init__(self):
        super(Prey, self).__init__()
        self.death_timer = 0

    @property
    def p_pos(self):
        return self._x, self._y

    def reset(self):
        super(Prey, self).reset()
        self.death_timer = 0

    def should_reincarnate(self):
        self.death_timer += 1
        return self.death_timer >= 15

class Scenario(BaseScenario):
    consumers = []

    def __init__(self):
        super().__init__()
        self.step = 0
        self.consumers = Scenario.consumers
        self.n_agents = 26
        self.lambda_value = 0.5
        self.predator = Predator()       
        self.encounters_all, self.encounters_step = [], []
        self.capture_times = np.zeros(n_predator, dtype=np.float64)  
        self.start_time = time.time()

    def make_world(self):
        world = World(width=map_size, height=map_size)

        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }

        for i in range(n_predator):
            agents.append(Predator(power_threshold=power_threshold, power_dec=power_dec))
            self.atype_to_idx["predator"].append(i)

        for i in range(n_prey):
            agents.append(Prey())
            self.atype_to_idx["prey"].append(n_predator + i)

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1

        self.reset_world(world)
        self.world = world
        return world

    def reward(self, agent, world):
        if agent == world.agents[0]:
            self.step += 1
            self.prey_captured = 0
            self.consumers = []
            for i in self.atype_to_idx["prey"]:
                prey = world.agents[i]

                if not prey.exists:
                    if prey.should_reincarnate():
                        prey.reset()
                        world.placeObj(prey)
                        continue

                if prey.exists and prey.captured:
                    world.removeObj(prey)
                    self.consumers.extend(prey.consumers)
                    self.prey_captured += 1

                    for predator_id in prey.consumers:
                        if predator_id < n_predator:
                            if self.capture_times[predator_id] == 0:  
                                self.capture_times[predator_id] = time.time() - self.start_time
                
        involved = self.consumers.count(agent.id)
        return agent.base_reward(self.prey_captured, involved, (self.step == max_step_per_ep))

    def done(self, agent, world):
        return (self.step == max_step_per_ep)

    def encounters_count(self):
        return self.encounters_all, self.encounters_step 
    
    def get_capture_times(self):
        return self.capture_times