import configparser
from instance_generator import EnvConfig, InstanceConfig, GenConfig, RLConfig
import Utils


class Parser(object):
    def __init__(self, parser):
        configp = configparser.ConfigParser()
        configp.read("config.ini")

        parser.add_argument('--operation', nargs="+", help="train or test?", type=str, metavar="operation")
        parser.add_argument('--density', nargs="?", help="density class of instances", type=int,
                            metavar="density_class")
        parser.add_argument('--q', nargs="+", help="capacity of vehicle", type=int, metavar="capacity")
        parser.add_argument('--dl', nargs="?", help="duration limit of vehicle", type=float, metavar="dl")
        parser.add_argument('--trials', nargs="?", help="number of trials to train", type=int, metavar="trials",
                            default=100000)
        parser.add_argument('--base_address', nargs="?", help="base_address", type=str, metavar="base_address",
                            default="Models/")
        parser.add_argument('--instance_count', nargs="?", help="number of instances", type=int,
                            metavar="instance_count")
        parser.add_argument('--nb', nargs="?", help="neighbor customers", type=int, metavar="nb", default=15)
        parser.add_argument('--start_train', nargs="?", help="start train trial", type=int, metavar="start_train",
                            default=0)
        parser.add_argument('--code', nargs="?", help="code", type=str, metavar="code",
                            default="")

        args = parser.parse_args()

        # Instances
        # self.n_customers = args.c[0] if args.c is not None else 0
        # self.n_vehicles = args.v[0] if args.v is not None else 0
        self.capacity = args.q
        self.duration_limit = args.dl
        # self.stoch_type = args.sv
        self.density_class = args.density
        # self.instance_class = args.instance_class

        # General simulator
        # self.model_type = args.model[0]
        self.operation = args.operation[0]
        self.trials = args.trials
        self.code = args.code
        self.base_address = args.base_address
        self.nb = args.nb
        self.instance_count = args.instance_count
        self.start_train_trial = args.start_train
        # self.generalized = args.generalized
        # self.preempt_action = args.preempt_action

        self.env_config = configp["Environment"]
        self.rl_config = configp["RL"]

    def get_env_config(self):
        env_args = {"service_area": Utils.str_to_arr(self.env_config["service_area"])}
        env_config = EnvConfig(**env_args)
        return env_config

    def get_instance_config(self):
        instance_args = {"capacity": self.capacity,
                         "density_class": self.density_class,
                         "depot": Utils.str_to_arr(self.env_config["depot"])}
        return InstanceConfig(**instance_args)

    def get_general_config(self):
        general_config = {"operation": self.operation,
                          "trials": self.trials,
                          "code": self.code,
                          "base_address": self.base_address,
                          "nb": self.nb,
                          "instance_count": self.instance_count,
                          "start_train_trial": self.start_train_trial}
        return GenConfig(**general_config)

    def get_rl_config(self):
        nn = {"nb": self.nb,
              "trials": self.trials}
        for k, v in self.rl_config.items():
            if k == "lr_decay":
                nn[k] = Utils.str_to_arr(v)
            elif k in ["gama", "lr", "update_prob"]:
                nn[k] = float(v)
            else:
                nn[k] = int(v)
        return RLConfig(**nn)
