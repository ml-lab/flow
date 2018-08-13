from flow.core.experiment import SumoExperiment
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.utils.rllib import get_rllib_config, get_flow_params
from flow.utils.registry import make_create_env

from flow.benchmarks.grid0 import flow_params as grid0
from flow.benchmarks.grid1 import flow_params as grid1
from flow.benchmarks.bottleneck0 import flow_params as bottleneck0
from flow.benchmarks.bottleneck1 import flow_params as bottleneck1
from flow.benchmarks.bottleneck2 import flow_params as bottleneck2
from flow.benchmarks.figureeight0 import flow_params as figureeight0
from flow.benchmarks.figureeight1 import flow_params as figureeight1
from flow.benchmarks.figureeight2 import flow_params as figureeight2
from flow.benchmarks.merge0 import flow_params as merge0
from flow.benchmarks.merge1 import flow_params as merge1
from flow.benchmarks.merge2 import flow_params as merge2

import ray
from ray.rllib.agent import get_agent_class
from ray.tune.registry import get_registry, register_env
import numpy as np
import joblib

# number of simulations to execute when computing performance scores
NUM_RUNS = 100

# dictionary containing all available benchmarks and their meta-parameters
AVAILABLE_BENCHMARKS = {"grid0": grid0,
                        "grid1": grid1,
                        "bottleneck0": bottleneck0,
                        "bottleneck1": bottleneck1,
                        "bottleneck2": bottleneck2,
                        "figureeight0": figureeight0,
                        "figureeight1": figureeight1,
                        "figureeight2": figureeight2,
                        "merge0": merge0,
                        "merge1": merge1,
                        "merge2": merge2}


def evaluate_policy(benchmark, compute_rl_actions, get_state=None):
    """Evaluates the performance of a controller on a predefined traffic
    benchmark.

    Parameters
    ----------
        benchmark : str
            name of the benchmark, must be printed as it is in the
            benchmarks folder; otherwise a ValueError will be raised
        compute_rl_actions : method
            the mapping from states to actions for the RL agent(s)
        get_state : method, optional
            a mapping from the environment object in Flow to some state, which
            overrides the get_state method of the environment. Note that the
            same cannot be done for the actions.

    Returns
    -------
        float
            mean of the evaluation return of the benchmark from NUM_RUNS number
            of simulations
        float
            standard deviation of the evaluation return of the benchmark from
            NUM_RUNS number of simulations

    Raises
    ------
        ValueError
            If the specified benchmark is not available.
    """
    if benchmark not in AVAILABLE_BENCHMARKS.keys():
        raise ValueError("benchmark {} is not available. Check spelling?".
                         format(benchmark))

    # get the flow params from the benchmark
    flow_params = AVAILABLE_BENCHMARKS[benchmark]

    exp_tag = flow_params["exp_tag"]
    sumo_params = flow_params["sumo"]
    vehicles = flow_params["veh"]
    env_params = flow_params["env"]
    env_params.evaluate = True  # Set to true to get evaluation returns
    net_params = flow_params["net"]
    initial_config = flow_params.get("initial", InitialConfig())
    traffic_lights = flow_params.get("tls", TrafficLights())

    # import the environment, scenario, and generator classes
    module = __import__("flow.envs", fromlist=[flow_params["env_name"]])
    env_class = getattr(module, flow_params["env_name"])
    module = __import__("flow.scenarios", fromlist=[flow_params["scenario"]])
    scenario_class = getattr(module, flow_params["scenario"])
    module = __import__("flow.scenarios", fromlist=[flow_params["generator"]])
    generator_class = getattr(module, flow_params["generator"])

    # recreate the scenario and environment
    scenario = scenario_class(name=exp_tag,
                              generator_class=generator_class,
                              vehicles=vehicles,
                              net_params=net_params,
                              initial_config=initial_config,
                              traffic_lights=traffic_lights)

    env = env_class(env_params=env_params,
                    sumo_params=sumo_params,
                    scenario=scenario)

    # make sure the get_state method of the environment is the one
    # specified by the user
    if get_state is not None:
        env.get_state = get_state

    # create a SumoExperiment object with the "rl_actions" method as
    # described in the inputs. Note that the state may not be that which is
    # specified by the environment.
    exp = SumoExperiment(env=env, scenario=scenario)

    # run the experiment and return the reward
    res = exp.run(num_runs=NUM_RUNS, num_steps=env.env_params.horizon,
                  rl_actions=compute_rl_actions)

    return np.mean(res["returns"]), np.std(res["returns"])


def get_compute_action_rllab(path_to_pkl):
    """Collects the compute_action method from rllab's pkl files.

    Parameters
    ----------
        path_to_pkl : str
            pkl file created by rllab that contains the policy information

    Returns
    -------
        method
            the compute_action method from the algorithm along with the trained
            parameters
    """
    # get the agent/policy
    data = joblib.load(path_to_pkl)
    agent = data['policy']

    # restore the trained parameters
    agent.restore()

    # the compute action return an action and an info_dict, so modify to just
    # return the action
    def compute_action(state):
        return agent.compute_action(state)[0]

    return compute_action


def get_compute_action_rllib(path_to_dir, checkpoint_num, alg):
    """Collects the compute_action method from RLlib's serialized files.

    Parameters
    ----------
        path_to_dir : str
            RLlib directory containing training results
        checkpoint_num : int
            checkpoint number / training iteration of the learned policy
        alg : str
            name of the RLlib algorithm that was used during the training
            procedure

    Returns
    -------
        method
            the compute_action method from the algorithm along with the trained
            parameters
    """
    # collect the configuration information from the RLlib checkpoint
    result_dir = path_to_dir if path_to_dir[-1] != '/' else path_to_dir[:-1]
    config = get_rllib_config(result_dir)

    # run on only one cpu for rendering purposes
    ray.init(num_cpus=1)
    config["num_workers"] = 1

    # create and register a gym+rllib env
    flow_params = get_flow_params(config)
    create_env, env_name = make_create_env(params=flow_params, version=9999,
                                           sumo_binary="sumo")
    register_env(env_name, create_env)

    # recreate the agent
    agent_cls = get_agent_class(alg)
    agent = agent_cls(env=env_name, registry=get_registry(), config=config)

    # restore the trained parameters into the policy
    checkpoint = result_dir + '/checkpoint-{}'.format(checkpoint_num)
    agent._restore(checkpoint)

    return agent.compute_action
