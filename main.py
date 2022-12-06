import argparse
import random
import tensorflow as tf
import Utils
import instance_generator
import myparser
import vrp
import vrpsd_solver


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    parser = myparser.Parser(p)
    normalize_instances = True

    env_config = parser.get_env_config()
    if normalize_instances:
        env_config.service_area[0] /= Utils.Norms.COORD
        env_config.service_area[1] /= Utils.Norms.COORD

    instance_config = parser.get_instance_config()
    gen_config = parser.get_general_config()
    rl_config = parser.get_rl_config()

    # generate/load instances
    instances = None

    instance_config.depot = [50, 50]
    env_config.service_area = [100, 100]
    capacity_list = [parser.capacity[0] / Utils.Norms.Q]
    if parser.operation == "test":
        instances = instance_generator.load_vrpvc_instances(f"{parser.density_class}_{parser.capacity[0]}")
    else:
        instances = instance_generator.generate_vrpscd_instances_generalized(instance_config=instance_config,
                                                                             density_class_list=[parser.density_class],
                                                                             capacity_list=capacity_list,
                                                                             count=100)
        dls_list = set([i["Config"].duration_limit for i in instances])

    # Initialize the environment
    vrpsd = vrp.VRPVCPO(env_config)

    if parser.operation == "train":
        caps = "".join([str(m) for m in instance_config.capacity])

        gen_config.code = f"{instance_config.density_class}_{caps}_" \
                          f"{random.randint(1000, 9999)}"

        print(f"Model code is {gen_config.code}")
        print("Params:")
        print("general config", gen_config)
        print("instance config", instance_config)
        print("rl config", rl_config)
        print("env config", env_config)

        with tf.Session() as sess:
            learner = vrpsd_solver.Learner(env=vrpsd, instances=instances, test_instance=instances[0],
                                           rl_config=rl_config, gen_config=gen_config, sess=sess)
            print("Trials\t#trains\treward\ttime\tloss")
            results = learner.train()
            learner.save_model()
        print("Done!")
    elif parser.operation == "train":
        vrpsd = vrp.VRPVCPO(env_config)
        caps = "".join([str(m) for m in instance_config.capacity])
        # gen_config.code = f"{instance_config.density_class}_{caps}_" \
        #                   f"{random.randint(1000, 9999)}"

        print(f"Model code is {gen_config.code}")
        print("Params:")
        print("general config", gen_config)
        print("instance config", instance_config)
        print("rl config", rl_config)
        print("env config", env_config)

        with tf.Session() as sess:
            learner = vrpsd_solver.Learner(env=vrpsd, instances=instances, test_instance=instances[0],
                                           rl_config=rl_config, gen_config=gen_config, sess=sess)
            learner.load_model()
            print("Trials\t#trains\treward\ttime\tloss")
            results = learner.train()
            learner.save_model()
        print("Done!")

    elif parser.operation == "test_min":
        with tf.Session() as sess:
            vrpsd = vrp.VRPVCPO(env_config)
            learner = vrpsd_solver.Learner(env=vrpsd, instances=instances, test_instance=instances[0],
                                           rl_config=rl_config, gen_config=gen_config, sess=sess)
            learner.load_model()
            rr = 0
            for e, instance in enumerate(instances[rr:rr + 1]):

                # if scenarios_set is not None:
                #     scenarios = np.array(scenarios_set[e]) / Utils.Norms.Q
                #     # scenarios = scenarios[:5]
                # else:
                #     scenarios = None
                # avg_rew = 0
                # for scenario in scenarios:
                #     res = vrp_sim.simulate(instance, scenario, method="random")
                #     avg_rew += res.final_reward
                # avg_rew /= len(scenarios)

                avg_rew = learner.test(instance, visualize=True)
                # print(avg_rew)

                print(f"{e + 1}\t{instance['Config'].real_n}\t"
                      # f"{instance['Config'].stoch_type}", 
                      f"{avg_rew}")

    else:
        print("Operation is not defined.")

