import math
import random
import numpy as np
import tensorflow as tf
import instance_generator
import simulated_annealing
import Utils
import rl
import vrp


class Learner:
    def __init__(self, env, instances, test_instance, rl_config, gen_config, sess):
        self.instances = instances
        self.env = env
        self.env.initialize_environment(instances[0])
        self.test_instance = test_instance
        self.max_trials = gen_config.trials
        self.gen_config = gen_config
        self.trials = gen_config.start_train_trial
        rl_config.max_trials = self.max_trials
        self.model = rl.QLearning(env, rl_config, sess, feat_size_c=8, feat_size_v=5)
        self.code = self.gen_config.code

    def train(self):
        rl_config = self.model.rl_config
        update_prob = rl_config.update_prob

        fr_list = []
        learning_loss = [0]
        n_period_result = 100

        counter = 0
        self.model.stop = False
        update_counter = 0
        # n_step_q = self.model.rl_config.q_steps
        # gradient_record = []

        lr_decay_params = rl_config.lr_decay

        lr_decay = Utils.LinearSchedule(init_t=self.gen_config.trials // 10,
                                        end_t=2 * self.gen_config.trials // 3,
                                        init_val=lr_decay_params[2], end_val=lr_decay_params[3],
                                        update_every_t=lr_decay_params[4])
        new_lr = lr_decay.init_val
        hp_decay = Utils.LinearSchedule(init_t=self.gen_config.trials // 5, end_t=2 * self.gen_config.trials // 3,
                                        init_val=1., end_val=.25, update_every_t=30000)

        self.model.dqn.set_opt_param(new_lr=new_lr, new_hp=hp_decay.init_val, sess=self.model.sess)

        prev_avg_rewards = 0

        test_scenarios = None

        # save the default models
        self.model.save_network(self.gen_config.base_address + self.code, self.code)
        self.model.save_network(self.gen_config.base_address + self.code + "/final", self.code)

        time_tracker = Utils.TimeTracker()
        while not self.model.stop:
            experience_buffer = []

            # update the pool of instances
            if self.trials % 20000 == 0:
                # generate 50 new instances and replace them in instances
                new_instances = instance_generator.generate_vrpscd_instances_generalized(
                    instance_config=self.env.ins_config,
                    density_class_list=[self.env.ins_config.density_class],
                    capacity_list=[self.env.ins_config.capacity],
                    count=50)

                # remove old ones
                old_codes = [m["Name"] for m in self.instances[:50]]
                del self.instances[:50]
                self.instances.extend(new_instances)
                for o in old_codes:
                    self.env.all_distance_tables.pop(o, None)

            ins_id = math.floor(random.random() * len(self.instances))
            instance = self.instances[ins_id]

            self.model.env.reset(instance=instance, scenario=None)

            # customer selection process
            self.model.customer_set_selection()

            # time scheduler
            for j in range(self.env.ins_config.m):
                self.model.env.TT.append((j, 0))

            self.model.env.final_cost = 0

            decision_epoch_counter = 0

            # consider using heapq instead of sorting the TT
            while len(self.model.env.TT) > 0:
                self.model.env.TT.sort(key=lambda x: x[1])
                v, stime = self.model.env.TT.pop(0)
                self.model.env.time = stime

                # updates the state when the vehicle arrives at a destination
                self.model.env.state_transition(v)

                # select action
                obs_k, x_k, preemptive, is_terminal, selected_index = self.model.choose_action(v, self.trials,
                                                                                               train=True)
                cost_to_learn, t_k = self.model.env.post_decision(x_k, v, preemptive)

                self.model.env.actions[v].append(x_k)

                if len(experience_buffer) > rl_config.q_steps - 1:
                    tmpr = sum([r[2] * (rl_config.gama ** ii) for ii, r in enumerate(experience_buffer)])
                    tmp_exp = experience_buffer.pop(0)
                    tmp_exp[2] = tmpr
                    tmp_exp.append(obs_k)
                    self.model.memory.add_sample(tmp_exp)
                experience_buffer.append([obs_k, selected_index, cost_to_learn, is_terminal])

                # schedule the next event for vehicle v if it is not terminated
                if not is_terminal:
                    self.model.env.TT.append((v, t_k))
                else:
                    # check if there is nothing is scheduled, set the last experience as the terminal
                    if len(self.model.env.TT) == 0:
                        experience_buffer[-1][-1] = 1

                self.model.env.final_cost += cost_to_learn
                decision_epoch_counter += 1

                if random.random() < update_prob and self.model.memory.get_size() > 1000:
                    loss, gradient = self.model.learn(self.trials)

                    learning_loss.append(loss)
                    # gradient_record.append(gradient)
                    update_counter += 1

                    if len(learning_loss) > 500:
                        learning_loss.pop(0)
                        # gradient_record.pop(0)

            self.model.DecisionEpochs += decision_epoch_counter

            # end of trial
            # self.model.env.time = self.env.ins_config.duration_limit
            # experience_buffer.append([obs_k, 0, 0, 1])

            while len(experience_buffer) > 0:
                tmpr = sum([r[2] for r in experience_buffer])
                tmp_exp = experience_buffer.pop(0)
                tmp_exp[2] = tmpr
                tmp_exp.append(obs_k)
                if tmp_exp[0] is not None and obs_k is not None:
                    self.model.memory.add_sample(tmp_exp)

            fr_list.append(self.model.env.final_cost)

            self.model.TrainTime += time_tracker.timeit()

            # update learning rate
            if lr_decay.update_time(self.trials):
                new_lr = lr_decay.val(self.trials)
                self.model.dqn.set_opt_param(new_lr=new_lr, sess=self.model.sess)

            # update huber parameter
            if hp_decay.update_time(self.trials):
                new_hp = hp_decay.val(self.trials)
                self.model.dqn.set_opt_param(new_hp=new_hp, sess=self.model.sess)

            self.trials += 1

            if self.trials % n_period_result == 0:
                # avg_final_reward = np.mean(fr_list) * self.env.ins_config.capacity
                avg_final_costs = np.mean(fr_list)
                print(f"{self.trials}\t{update_counter}\t{avg_final_costs:.2f}\t{self.model.TrainTime:.1f}\t"
                      f"{np.mean(learning_loss):.6f}")
                fr_list = []
                self.model.zero_q = 0

            if self.trials % self.model.rl_config.test_every == 0:
                if test_scenarios is None:
                    # generate test scenarios
                    test_scenarios = [[vrp.demand_realization(c[3])
                                      for c in self.test_instance["Customers"]]
                                      for _ in range(100)]

                c_res = Utils.TestResults()

                for test_scenario in test_scenarios:
                    res = self.model.test_model(self.test_instance, test_scenario)
                    c_res.accumulate_results(res)
                c_res.print_avgs_full()
                avg_final_reward = c_res.get_avg_final_reward()
                if avg_final_reward > prev_avg_rewards:
                    prev_avg_rewards = avg_final_reward
                    self.model.save_network(self.gen_config.base_address + self.gen_config.code, self.gen_config.code,
                                            False)
                self.model.zero_q = 0

            # save the model every
            if self.trials % 1000 == 0:
                saving_code = self.gen_config.code + "/final"

                self.model.save_network(self.gen_config.base_address + saving_code,
                                        saving_code, False)

            counter += 1
            # stoppage criteria
            if self.trials > self.max_trials:
                self.model.stop = True

        return None

    def save_model(self):
        self.model.save_network(self.gen_config.base_address + self.gen_config.code + "/final",
                                self.gen_config.code)

    def load_model(self):
        new_saver = tf.compat.v1.train.Saver()
        # model_dir = f"{self.gen_config.base_address}{self.gen_config.code}"
        # model_dir = f"{self.gen_config.base_address}{self.gen_config.code}/final"
        model_dir = f"{self.gen_config.base_address}{self.gen_config.code}/final/{self.gen_config.code}"

        new_saver.restore(self.model.sess, tf.train.latest_checkpoint(model_dir))

    def test(self, instance, visualize=False):
        c_res = Utils.TestResults()

        if visualize:
            for _ in range(3):
                res = self.model.test_model(instance, None)
                Utils.plot_environment(c=self.model.env.customers,
                                       v=res.actions,
                                       depot=[.50, .50],
                                       service_area_length=[1.00, 1.00],
                                       detail_level=3,
                                       animate=False)
                res.print_full()
                print(res.agent_record)
                res.print_actions()

        else:
            for _ in range(500):
                res = self.model.test_model(instance, None)
                c_res.accumulate_results(res)

            # c_res.print_avgs_full()
            return c_res.get_avg_final_reward()

    # def evaluate_solution(self, selected_customers):

        # obs, (target_customers_dir, target_customers_indir), is_terminal = self.model.observation_function(k)

        # return 0

    # def simulated_annearling_runner(self, instance):
    #     self.env.reset(instance, scenario=None, reset_distance=True)
    #
    #     # initialize x_c, x_b
    #     init_solution = simulated_annealing.greedy_solution(instance)
    #     init_value = self.evaluate_solution(init_solution)
    #
    #     x_c = list(init_solution)
    #     x_b = list(init_solution)
    #
    #     val_c = init_value + 0.
    #     val_b = init_value + 0.
    #
    #     # set temperature
    #     temp = 100
    #     temp_min = 0.001
    #     kappa = 10.
    #     alpha = 0.95
    #
    #     # set iteration
    #     itr = 0
    #     itr_max = 100
    #
    #     # set neighbor count
    #     neighbors_count = 100
    #
    #     while temp > temp_min:
    #         while itr < itr_max:
    #             x_neighbors = [simulated_annealing.generate_neighbor(x_c, customers)
    #                            for _ in range(neighbors_count)]
    #             x_neighbors_values = [self.evaluate_solution(x) for x in x_neighbors]
    #             best_loc_id = np.argmin(x_neighbors_values)
    #             best_loc_x = x_neighbors[best_loc_id]
    #             best_loc_value = x_neighbors_values[best_loc_id]
    #
    #             if best_loc_value < val_b:
    #                 x_c = list(best_loc_x)
    #                 x_b = list(best_loc_x)
    #
    #                 val_c = best_loc_value + 0.
    #                 val_b = best_loc_value + 0.
    #             else:
    #                 acceptance_prob = np.exp(-(best_loc_value - val_c) / (temp * kappa))
    #                 if random.Random() < acceptance_prob:
    #                     x_c = list(best_loc_x)
    #                     val_c = best_loc_value + 0.
    #             itr += 1
    #         temp *= alpha
    #     return x_b