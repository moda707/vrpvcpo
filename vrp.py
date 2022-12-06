import copy
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
import Utils


def demand_realization(exp_dem):
    if exp_dem > 0.1:
        d = ((random.random() * 10 - 5) / Utils.Norms.Q) + exp_dem
    else:
        d = ((random.random() * 8 + 1) / Utils.Norms.Q)
    if d > 0.4:
        print("asd")
    return d


class VRPVCPO(object):
    def __init__(self, env_config):
        self.vehicles = None
        self.customers = None
        self.c_enc = None
        self.v_enc = None

        self.time = 0
        self.env_config = env_config
        self.ins_config = None

        self.demand_scenario = None

        self.actions = {}
        self.TT = []
        self.final_reward = 0

        self.all_distance_tables = {}
        self.distance_table = None

        self.vehicles_traveled_time = []
        self.vehicles_traveled_time2 = []
        self.p_l = 5.

        self.vehicles_terminated = None

    def initialize_environment(self, instance):
        self.customers = np.array(instance["Customers"])
        self.vehicles = np.array(instance["Vehicles"])
        self.ins_config = instance["Config"]

        self.update_distance_table(instance["Name"])

    def reset(self, instance=None, scenario=None):
        self.time = 0
        self.customers = np.zeros([self.ins_config.n + 2, 9])
        self.customers[:-2, :5] = instance["Customers"]
        self.customers[:-2, -1] = instance["Customers"][:, -2]
        self.customers[:-2, -2] = 1
        customers_condition = self.customers[:, 4] == -1
        self.customers[customers_condition, 5] = 1
        self.customers[customers_condition, 7] = 0
        self.customers[-2] = np.array([self.ins_config.depot[0], self.ins_config.depot[1], 0., 0., 0., 0., 1., 0., 0.])
        self.customers[-1] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])

        self.vehicles = np.array(instance["Vehicles"])
        self.ins_config = copy.deepcopy(instance["Config"])

        self.update_distance_table(instance["Name"])

        self.demand_scenario = scenario

        self.actions = {}
        self.vehicles_traveled_time = []
        self.vehicles_traveled_time2 = []
        for m in range(self.ins_config.m):
            self.actions[m] = [-2]
            self.vehicles_traveled_time.append(0.)
            self.vehicles_traveled_time2.append(0.)

        self.vehicles_terminated = np.zeros(self.ins_config.m)

    def update_distance_table(self, instance_name):
        if instance_name in self.all_distance_tables:
            self.distance_table = squareform(self.all_distance_tables[instance_name])
        else:
            # generate distance table
            pos_list = list(self.customers[:, :2])
            pos_list.append(list(self.ins_config.depot))
            distance_table = pdist(np.array(pos_list))
            self.all_distance_tables[instance_name] = distance_table
            self.distance_table = squareform(distance_table)

    def post_decision(self, x, v, preemptive):
        # in this function, the current state transits to the post decision state.
        # it means, action x, only blocks customer x temporary to not be served by any other vehicles and

        # updates the position and the arrival time of the vehicle k
        depot = self.ins_config.depot
        n = self.ins_config.n
        v_k = self.vehicles[v]
        q = v_k[2]
        loc_id = int(v_k[4])

        if x == n:
            # capacity is zero
            if q == 0:
                tt = self.distance_table[loc_id][n]
            else:
                # it is the terminal. no other choice is available
                tt = 0
                self.vehicles_terminated[v] = 1
            psi = depot
            loc_id = n
        else:
            c = self.customers[x]
            psi_x = c[:2]

            if preemptive:
                tt = self.distance_table[loc_id][n] + self.distance_table[n][x]

                # restock
                q = self.ins_config.capacity
            else:
                tt = self.distance_table[loc_id][x]

            psi = psi_x
            # tag the customer to unavailable
            c[2] = 0
            loc_id = x

        at = self.time + tt
        # Update the V_k in Global state
        self.vehicles[v] = [psi[0], psi[1], q, at, loc_id]

        cost = tt
        if at > self.ins_config.duration_limit:
            cost += (self.p_l - 1.) * (at - max(self.ins_config.duration_limit, self.time))

        return cost, at

    def state_transition(self, v):
        n = self.ins_config.n
        v_k = self.vehicles[v]
        served_demand = 0
        loc_id = int(v_k[4])

        if loc_id == n:
            v_k[2] = self.ins_config.capacity

        else:
            # loc_id
            cur_cus = self.customers[loc_id]

            # if the demand is not realized yet, get a realization
            if cur_cus[4] == -1:
                if self.demand_scenario is None:
                    w = demand_realization(cur_cus[3])
                else:
                    w = self.demand_scenario[loc_id]

                cur_cus[4] = w
                cur_cus[-1] = w

            served_demand = min(cur_cus[4], v_k[2])
            cur_cus[4] -= served_demand
            cur_cus[-1] = cur_cus[4] + 0.
            v_k[2] -= served_demand

            cur_cus[2] = cur_cus[4] > 1e-5

        return served_demand

    def get_available_customers(self, v):
        v_k = self.vehicles[v]
        loc_id = int(v_k[4])

        is_terminal = 0

        if v_k[2] == 0:
            target_customers = []
        else:
            # distance i to j then j to depot
            avail_customers_cond = self.customers[:, 2] == 1
            target_customers = np.where(avail_customers_cond)[0].tolist()

            # if there is at leas one vehicle other than v (not terminated), allow v to terminate
            if self.ins_config.m - sum(self.vehicles_terminated) > 1.01:
                target_customers.append(self.ins_config.n)

        if loc_id == self.ins_config.n and len(target_customers) == 0:
            is_terminal = 1
        return target_customers, is_terminal
