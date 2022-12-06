import copy
import json
import random
import numpy as np
import Utils


def load_vrpvc_instances(fname):

    with open("Instances/VRPVC/" + fname, 'r') as f:
        s = json.load(f)
    random_instances = []

    for e, instance in enumerate(s):
        v_set = np.array(instance["Vehicles"])
        c_set = np.array(instance["Customers"])

        # Normalize
        v_set /= [Utils.Norms.COORD, Utils.Norms.COORD, Utils.Norms.Q, 1, 1]
        c_set /= [Utils.Norms.COORD, Utils.Norms.COORD, 1, Utils.Norms.Q]

        cn = instance["Config"]
        config = InstanceConfig(**cn)
        config.density_class = int(str.split(fname, "_")[0])

        config.depot[0] /= Utils.Norms.COORD
        config.depot[1] /= Utils.Norms.COORD
        config.duration_limit /= Utils.Norms.COORD

        c_set[config.real_n:, 0] = 0

        # config.real_n = len(c_set)
        # config.duration_limit = cn["duration_limit"]
        # config.m = cn["m"]
        # config.n = len(c_set)
        # config.capacity = cn["capacity"]
        # config = collections.namedtuple("InstanceConfig", cn.keys())(*cn.values())
        # config = ModelConfig(Q=cn["Q"], m=cn["m"], dl=cn["dl"], xy_length=cn["xy_length"])
        random_instances.append({"Vehicles": v_set, "Customers": c_set, "Config": config, "Name": e})

    return random_instances


def generate_vrpscd_instances_generalized(instance_config, density_class_list, capacity_list, count,
                                          max_c_size=None, max_v_size=None):
    """
    Notes:
    1-since the number of realized customers is random, we define the vector of customers with a fixed size equal to
    1.2*|nbar|. In case of having fewer realized customers, we fill the vector with dummy customers.
    2- the last index in the set of customers (nodes) always refers to the depot.

    :param config: the characteristics of the environment
    :param density_class: the density level of customers (low, moderate, and high)
    :param n_vehicles: the number of vehicles
    :param duration_limit: the duration limit
    :param capacity: the max capacity of vehicles
    :param count: the number of instances to be generated
    :return: a set of instances
    """
    # use big heatmaps to generate instances
    heatmap = [[1, 1, 0, 1, 0],
               [1, 1, 0, 0, 1],
               [1, 0, 1, 1, 0],
               [0, 1, 0, 1, 1],
               [0, 1, 1, 1, 0]]
    instances = []

    nbar_list = [23, 53, 83]
    m_list = [3, 7, 11]
    L_list = [221.47, 195.54, 187.29, 9999999]
    n_probs = [0.1, 0.4, 0.4, 0.1]
    n_numbers_list = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]
    #   the distribution function for the expected demands (uniform)
    exp_demands = [5, 10, 15]

    if max_c_size is None:
        max_c_size = int(1.2 * nbar_list[int(max(density_class_list))])
    if max_v_size is None:
        max_v_size = m_list[int(max(density_class_list))]

    for _ in range(count):
        density_class = random.choice(density_class_list)
        capacity = random.choice(capacity_list)

        #   set the instance config
        config = copy.deepcopy(instance_config)

        config.capacity = capacity
        config.duration_limit = L_list[density_class] / Utils.Norms.COORD
        config.real_duration_limit = config.duration_limit + 0.
        config.m = int(m_list[density_class] + 0.)
        nbar = int(nbar_list[density_class])
        n_numbers = n_numbers_list[density_class]
        if config.depot[0] > 1:
            config.depot[0] /= Utils.Norms.COORD
            config.depot[1] /= Utils.Norms.COORD

        #   the set of vehicles [l_x, l_y, q, a, occupied_node]
        v_set = np.zeros([max_v_size, 5])
        # vehicles index 0, 1, ..., m-1
        for j in range(config.m):
            v_set[j] = [config.depot[0], config.depot[0],
                        config.capacity, 0, max_c_size]

        il = len(heatmap)
        jl = len(heatmap[0])

        # Generate a set of customers (location, availability, expected demand, unserved demand)
        c_set = np.zeros([max_c_size, 5])
        realized_pos = [(50, 50)]
        c_count = 0
        n_cust_limit = min(nbar * 1.2, max_c_size)
        #   enumerate over partitions of the heatmap and generate a random number of customers for eah that is active.
        for i in range(il):
            #   make sure the number of realized customers does not exceed the nbar.
            if c_count >= n_cust_limit:
                break

            for j in range(jl):

                if heatmap[i][j] == 0:
                    continue
                if c_count >= n_cust_limit:
                    break

                #   generate n_z
                n_z = np.random.choice(n_numbers, 1, p=n_probs)[0]
                for c in range(n_z):
                    x_coord = random.randint(j * 20 + 1, (j + 1) * 20)
                    y_coord = random.randint(i * 20 + 1, (i + 1) * 20)

                    #   make sure no two customers request from exactly the same location
                    while (x_coord, y_coord) in realized_pos:
                        x_coord = random.randint(j * 20 + 1, (j + 1) * 20)
                        y_coord = random.randint(i * 20 + 1, (i + 1) * 20)
                    realized_pos.append((x_coord, y_coord))

                    #   randomly assign an expected demand to the realized location
                    demand = random.choice(exp_demands) / Utils.Norms.Q

                    #   construct the customers raw feature set
                    # [l_x, l_y, h, d, dhat]
                    c_set[c_count] = [x_coord / Utils.Norms.COORD, y_coord / Utils.Norms.COORD, 1,
                                      demand, -1]
                    c_count += 1

                    if c_count >= n_cust_limit:
                        break

        config.n = len(c_set)
        config.real_n = c_count
        inst_name = "I_" + str(density_class) + "_" + str(config.m) + "_" + str(config.capacity) + "_" + \
                    str(random.randint(100000, 999999))
        instance = {"Vehicles": np.array(v_set), "Customers": np.array(c_set).astype(float), "Config": config,
                    "Name": inst_name}
        instances.append(instance)

    return instances


class InstanceConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


class EnvConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


class GenConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


class RLConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


