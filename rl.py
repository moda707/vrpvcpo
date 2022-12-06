import math
import random
import numpy as np
import tensorflow as tf
import Utils
import qnetwork
import vrp


class QLearning(object):
    def __init__(self, env: vrp.VRPVCPO, rl_config, sess, feat_size_c=8, feat_size_v=4):
        # set the environment
        self.env = env
        # self.ins_config = self.env.ins_config
        self.env_config = self.env.env_config

        # active fine-tuning?
        fine_tune = False

        # rl configs
        self.rl_config = rl_config
        self.lr = float(rl_config.lr)
        self.test_every = int(rl_config.test_every)
        self.ep_max = int(rl_config.ep_max)
        self.update_freq = int(rl_config.update_freq)
        self.replace_target_iter = int(rl_config.replace_target_iter)
        self.batch_size = int(self.rl_config.batch_size)
        self.memory_size = int(self.rl_config.memory_size)
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v

        self.memory = Memory(self.memory_size)
        self.replay_start = int(self.memory_size)
        self.dqn = None
        self.nb = rl_config.nb

        # loging
        self.depot_stay_count = []
        self.TrainTime = 0
        self.DecisionEpochs = 0
        self.epsilon = 1.
        self.update_counter = 0
        self.depot_stay_count = []
        self.zero_q = 0

        # epsilon decaying parameters
        self.main_variables = None
        self.build_net()
        self.sess = sess
        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(var_list=self.main_variables)

        # epsilon decaying strategy - parameters
        self.eps_A = 0.15
        self.eps_B = 0.1
        self.eps_C = 0.05
        self.Max_trials = rl_config.trials

    def build_net(self):
        dqn_args = {"init_lr": 0.001, "emb_size": 128, "feat_size_c": self.feat_size_c,
                    "feat_size_v": self.feat_size_v, "max_n": self.env.ins_config.n + 2}
        model = qnetwork.DQN(**dqn_args)

        self.dqn = model

    def observation_function(self, v):
        n = self.env.ins_config.n

        # take all feasible customers
        target_customers, is_terminal = self.env.get_available_customers(v)

        c_set = np.array(self.env.customers)[:, :-1]
        v_set = np.array(self.env.vehicles)[:, :]
        v_set[:, 3] -= self.env.time

        # the last component indicates whether the vehicle is terminated (1) or not (0)
        v_set[:, -1] = 0
        v_set[self.env.vehicles_terminated.astype(bool), -1] = 1
        v_set[self.env.vehicles_terminated.astype(bool), -2] = -1

        available_customers = np.zeros(n + 2)
        available_customers[:n] = self.env.customers[:n, 2] > 0

        obs = {"customers": c_set, "vehicles": v_set, "active_vehicle": v + 0,
               "time": [self.env.ins_config.duration_limit - self.env.time],
               "available_customers": available_customers}

        return obs, target_customers, is_terminal

    def choose_action(self, k, trials, train=True):
        """
        The action set is travelling to all customers directly or indirectly + the depot.
        Some of them can be infeasible.
        target_customers_dir, target_customers_indir: available actions
        For each customer id in o_target_customers, the dqn generates two q values for dir and indir visit.
        That is, the output is a vector of size 2 * nb
        Available_targets is a binary list that represents which elements of the output are available/feasible to choose
        Although the action for the vehicle is a customer from o_target_customers, the index of its action to train the
        dqn is an index from available_targets.

        Args:
            k:
            trials:
            train:

        Returns:

        """
        nb = self.env.ins_config.n + 1
        n = self.env.ins_config.n

        exp_coef = 16 / self.rl_config.max_trials
        if train:
            # epsilon = 1 - (0.9 * trials) / self.ep_max
            # epsilon = max(epsilon, 0.1)
            epsilon = np.exp(-trials * exp_coef) + 0.05 + 0.03 * (1 - trials / self.rl_config.max_trials)

        else:
            epsilon = 0.

        obs, target_customers, is_terminal = self.observation_function(k)
        n_actions_dir = len(target_customers)

        # target_customers always has at least one item
        if n_actions_dir == 0:
            # depot is the only choice to travel
            # it means there is only one choice, so take it.
            best_action = n
            preemptive = False
            selected_index = 0

            obs["available_targets"] = np.zeros(2 * nb)
            obs["available_targets"][-2] = 1

        else:
            # a binary vector representing target customers to approximate q values for them.
            # n+1 refers to a dummy customer to fill the list.
            o_target_customers = np.zeros(nb)
            o_target_customers[target_customers] = 1

            available_targets = np.zeros(2*nb)
            available_targets[np.array(target_customers) * 2] = 1
            if int(self.env.vehicles[k, -1]) != n:
                available_targets[np.array(target_customers) * 2 + 1] = 1
                # preemptive to the depot is not possible.
                available_targets[-1] = 0

            obs["available_targets"] = available_targets

            if random.random() <= epsilon:
                # explore
                sind = math.floor(random.random() * n_actions_dir)
                best_action = target_customers[sind]

                if random.random() < 0.5:
                    selected_index = best_action * 2
                    preemptive = False
                else:
                    selected_index = best_action * 2 + 1
                    preemptive = True

            else:
                obs_mk = {}
                for e1, e2 in obs.items():
                    obs_mk[e1] = np.expand_dims(e2, axis=0)
                obs_mk["active_vehicle"] = [[0, obs["active_vehicle"]]]

                q_values = self.dqn.value(obs=obs_mk, sess=self.sess)[0]
                q_values[q_values < 0.] = 0.
                q_values += (1 - available_targets) * 10e9

                selected_index = np.argmin(q_values)

                best_action = math.floor(selected_index / 2.)
                if selected_index % 2. == 1:
                    preemptive = True
                else:
                    preemptive = False

        if best_action == n and self.env.vehicles[k][2] > 0.0001:
            is_terminal = 1

        return obs, int(best_action), preemptive, is_terminal, selected_index

    def learn(self, trials):
        """
                each batch contains:
                1- obs: C, V, time, vb, act veh position, target_customers, available_targets, available customers, inst chars
                2- selected_target (action, an index in available targets): ind -> batch,ind
                3- reward
                4- is_terminal
                5- obs_next: same items as obs

                Procedure:
                - a batch of experiences is sampled from the memory
                - with double_qn, in order to compute the max_{x} Q(s_{k+1}, x), we use the primary network to decide the best,
                but evaluate the Q value of that action with the target network.
                - the future rewards are discounted considering the n-step q learning.

                """

        if trials % self.replace_target_iter == 0:
            self.sess.run(self.dqn.replace_target_op)

        batch = self.memory.sample(self.batch_size)

        def make_up_dict(o, axis=None):
            key_list = ["customers", "vehicles", "active_vehicle", "time",
                        "available_targets", "available_customers"]
            if axis is None:
                new_dict = dict((k, np.stack([v[k] for v in o])) for k in key_list)
            else:
                new_dict = dict((k, np.concatenate([v[k] for v in o], axis=axis)) for k in key_list)

            return new_dict

        batch_range = range(self.batch_size)

        obs = [val[0] for val in batch]
        obs = make_up_dict(obs)

        selected_targets = [val[1] for val in batch]
        rewards = [val[2] for val in batch]
        is_terminal = [val[3] for val in batch]

        # DQN is off-policy, so it needs to compute the max_a of the next state
        obs_next = [val[4] for val in batch]
        obs_next = make_up_dict(obs_next)

        obs["active_vehicle"] = np.array([[i, obs["active_vehicle"][i]] for i in batch_range])
        obs["selected_target"] = np.array([[i, selected_targets[i]] for i in batch_range])

        obs_next["active_vehicle"] = np.array([[i, obs_next["active_vehicle"][i]] for i in batch_range])

        # batch x nb
        next_q_values = self.dqn.value(obs=obs_next, sess=self.sess)
        # for not available targets, set the q value to a big positive number
        next_q_values += 1000 * (1 - obs_next["available_targets"])

        # if self.rl_config.use_double_qn:
        best_action = np.argmin(next_q_values, axis=1).reshape([-1, 1])

        target_q_values = self.dqn.value_(obs=obs_next, sess=self.sess)

        min_next_q_values = [target_q_values[i][best_action[i]] for i in batch_range]
        # else:
        #     max_next_q_values = np.max(next_q_values, axis=1)

        min_next_q_values = np.array(min_next_q_values).reshape(-1)
        min_next_q_values[min_next_q_values < 0] = 0.

        discount_factor = np.array([0. if m else self.rl_config.gama ** self.rl_config.q_steps
                                    for m in is_terminal])

        q_target = rewards + discount_factor * min_next_q_values

        loss, avg_gradient = self.dqn.optimize(obs=obs, target=q_target, sess=self.sess)
        return loss, avg_gradient

    def customer_set_selection(self, n=None):
        if n is None:
            # randomly remove 0% to 40% of customers
            n = int((random.random() * 40) * self.env.ins_config.real_n / 100.)
        m = 0
        while m < n:
            cn = int(random.random() * self.env.ins_config.real_n)
            if not (self.env.customers[cn][0] == 0 and self.env.customers[cn][1] == 0):
                m += 1
                self.env.customers[cn] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])

        self.env.ins_config.real_n -= m

    def test_model(self, test_instance, scenario, need_reset=True):
        sum_exp_served = 0
        n = self.env.ins_config.n
        m = self.env.ins_config.m

        self.env.actions = {}
        reset_distance = need_reset

        if test_instance is not None:
            self.env.reset(test_instance, scenario=scenario)
        else:
            self.env.reset(instance=test_instance, scenario=scenario)

        self.customer_set_selection(int(50 * self.env.ins_config.real_n / 100.))
        self.env.time_table = []
        for j in range(m):
            self.env.time_table.append((j, 0))

        # time scheduler
        agents_cost = [0] * m
        agents_record = {}
        agents_travel_time = [0] * m
        for j in range(m):
            self.env.actions[j] = [n]
            agents_record[j] = []

        final_cost = 0

        last_actions = {}
        n_routes = 0
        n_visits = 0
        n_preemptives = 0
        avg_terminal_time = 0
        max_travel_time = 0

        while len(self.env.time_table) > 0:
            self.env.time_table.sort(key=lambda x: x[1])
            v, time = self.env.time_table.pop(0)
            self.env.time = time

            # transit from s^x_{k-1} to s_k
            self.env.state_transition(v)

            # active vehicle v takes action
            _, x_k, preemptive, is_terminal, _ = self.choose_action(v, None, False)

            cost, t_k = self.env.post_decision(x_k, v, preemptive)

            if is_terminal:
                # record the duration of the trip for the terminated vehicle
                agents_travel_time[v] = time

            agents_cost[v] += cost
            agents_record[v].append(cost)

            if x_k != n:
                sum_exp_served += self.env.customers[x_k][-1]

            if preemptive:
                self.env.actions[v].append(-2)
                self.env.actions[v].append(x_k)
            else:
                self.env.actions[v].append(x_k)

            if x_k == n:
                if self.env.vehicles[v][-1] != n:
                    n_routes += 1
                    if self.env.vehicles[v][3] > 0 and not is_terminal:
                        n_preemptives += 1
            else:
                n_visits += 1

            if is_terminal == 1:
                avg_terminal_time += time
                max_travel_time = time

            last_actions[v] = x_k

            # schedule the next event for vehicle v if it still has time
            if not is_terminal:
                self.env.time_table.append((v, t_k))

            final_cost += cost

        # final_reward *= Utils.Norms.Q
        avg_terminal_time /= m
        n_served = len([c for c in self.env.customers if c[3] == 0])
        results = Utils.TestResults(final_cost=final_cost, actions=self.env.actions, n_routes=n_routes,
                                    n_served=n_served, avg_travel_time=avg_terminal_time, n_visits=n_visits,
                                    max_travel_time=max_travel_time, agents_cost=agents_cost)
        results.n_preemptives = n_preemptives
        if scenario is not None:
            results.tot_realized_demand = sum(scenario)
        results.service_rate = results.final_cost / results.tot_realized_demand
        results.agent_record = agents_record
        results.tot_travel_time = sum(agents_travel_time)

        return results

    def save_network(self, base_address, code, write_meta=True):
        # saver = tf.compat.v1.train.Saver()
        dir_name = base_address
        self.saver.save(self.sess, dir_name + "/" + code, write_meta_graph=write_meta)

    def epsilon_calculator(self, trial):
        standardized_time = (trial - self.eps_A * self.Max_trials) / (self.eps_B * self.Max_trials)
        cosh = np.cosh(math.exp(-standardized_time))
        epsilon = (1.05 + self.eps_C) - (1 / cosh + (trial * self.eps_C / self.Max_trials))

        # In transfer learning, it makes the simulation to exploit more.
        # epsilon /= 2.
        return epsilon

    def load_network(self, network_address):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(network_address))
        self.sess.run(self.dqn.replace_target_op)


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
        self._weights = []

    def add_sample(self, sample, weight=1):
        self._samples.append(sample)

        # self._weights.append(weight)

        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
            # self._weights.pop(0)

    def sample(self, no_samples):
        # if no_samples > len(self._samples):
        #     no_samples = len(self._samples)
        ll = self.get_size()
        out = [self._samples[int(random.random() * ll)] for _ in range(no_samples)]
        return out

    def sample_last(self, no_samples):
        if no_samples > len(self._samples):
            no_samples = len(self._samples)

        out = self._samples[-no_samples:]
        return out

    def get_size(self):
        return len(self._samples)
