import copy
import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer

# Import BEMAS reshaper
try:
    from components.bemas_reshaper import BEMASReshaper
except ImportError:
    BEMASReshaper = None


class BEMASQLearner:
    """Q-Learner with BEMAS: Per-agent, per-timestep reward reshaping."""
    
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.target_mac = copy.deepcopy(mac)

        # BEMAS reshaper
        self.use_bemas = getattr(args, 'use_bemas', False)
        if self.use_bemas:
            if BEMASReshaper is None:
                raise ImportError("BEMAS reshaper not found!")
            self.bemas = BEMASReshaper(args)
            self.logger.console_logger.info("BEMAS: Per-agent, per-timestep reshaping ENABLED")
        else:
            self.bemas = None

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]  # [bs, T-1, 1] or [bs, T-1, n_agents]
        actions = batch["actions"][:, :-1]  # [bs, T-1, n_agents, 1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # ============ BEMAS PER-TIMESTEP RESHAPING ============
        if self.use_bemas and self.bemas is not None:
            # Compute Q-values and policies for ALL timesteps
            mac_out_all = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out_all.append(agent_outs)
            mac_out_all = th.stack(mac_out_all, dim=1)  # [bs, T, n_agents, n_actions]
            
            # Reshape rewards for each batch, each timestep
            intrinsic_rewards_list = []
            for b in range(batch.batch_size):
                timestep_intrinsics = []
                for t in range(batch.max_seq_length - 1):  # T-1 timesteps
                    # Get data for this timestep
                    q_vals = mac_out_all[b, t]  # [n_agents, n_actions]
                    acts = actions[b, t, :, 0].long()  # [n_agents]
                    
                    # Convert Q-values to policies
                    policies = th.nn.functional.softmax(q_vals, dim=-1)
                    
                    # Compute intrinsic reward for this timestep
                    intrinsic = self.bemas.reshape_rewards_per_timestep(
                        q_vals, policies, acts, t_env
                    )
                    timestep_intrinsics.append(intrinsic)
                
                # Stack all timesteps for this batch
                intrinsic_rewards_list.append(th.stack(timestep_intrinsics, dim=0))  # [T-1, n_agents]
            
            intrinsic_rewards = th.stack(intrinsic_rewards_list, dim=0)  # [bs, T-1, n_agents]
            
            # Add intrinsic to environment rewards
            if self.args.common_reward:
                # rewards is [bs, T-1, 1]
                # Average intrinsic across agents for common setting
                intrinsic_common = intrinsic_rewards.mean(dim=-1, keepdim=True)
                rewards = rewards + intrinsic_common
            else:
                # rewards is [bs, T-1, n_agents] - add directly
                rewards = rewards + intrinsic_rewards
        # ====================================================

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1, "Expected singular dimension for common rewards"
            rewards = rewards.expand(-1, -1, self.n_agents)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [bs, T, n_agents, n_actions]
        
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate target Q-Values
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]


        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # TD-error
        td_error = chosen_action_qvals - targets.detach()

        # ============ UPDATE BEMAS TD SCORES ============
        if self.use_bemas and self.bemas is not None:
            # Compute per-agent TD-errors (before mixing)
            individual_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # [bs, T-1, n_agents]
            

            individual_targets = rewards  # Already includes intrinsic
            individual_td = (individual_qvals - individual_targets.detach()).abs()
        
            # Average TD-error per agent across batch and time
            per_agent_td = individual_td.mean(dim=[0, 1])  # [n_agents]
            
            # Update BEMAS TD-error based scores
            self.bemas.update_td_scores(per_agent_td)
        # ==============================================

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Loss
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and \
           (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.n_agents), t_env)
            
            # BEMAS logging
            if self.use_bemas and self.bemas is not None:
                alpha_t, beta_t = self.bemas.get_alpha_beta(t_env)
                self.logger.log_stat("bemas_alpha", alpha_t, t_env)
                self.logger.log_stat("bemas_beta", beta_t, t_env)
                self.logger.log_stat("bemas_curiosity_mean", self.bemas.curiosity_scores.mean().item(), t_env)
                self.logger.log_stat("bemas_curiosity_max", self.bemas.curiosity_scores.max().item(), t_env)
                self.logger.log_stat("bemas_curiosity_min", self.bemas.curiosity_scores.min().item(), t_env)
                self.logger.log_stat("bemas_performance_mean", self.bemas.performance_scores.mean().item(), t_env)
            
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

        if self.use_bemas and self.bemas is not None:
            self.bemas.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)

        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))