"""Visualizer for rllib experiments

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import os

from flow.utils.rllib import get_flow_params, get_rllib_config
from flow.utils.evaluate import get_compute_action_rllib
from flow.core.util import emission_to_csv
from flow.core.experiment import SumoExperiment

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO

Here the arguments are:
1 - the number of the checkpoint
PPO - the name of the algorithm the code was run with
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a reinforcement learning agent "
                "given a checkpoint.", epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument("result_dir", type=str,
                    help="Directory containing results")
parser.add_argument("checkpoint_num", type=str,
                    help="Checkpoint number.")

# optional input parameters
parser.add_argument("--run", type=str, default='PPO',
                    help="The algorithm or model to train. This may refer to "
                         "the name of a built-on algorithm (e.g. RLLib's DQN "
                         "or PPO), or a user-defined trainable function or "
                         "class registered in the tune registry.")
parser.add_argument('--num_rollouts', type=int, default=1,
                    help="The number of rollouts to visualize.")
parser.add_argument('--emission_to_csv', action='store_true',
                    help='Specifies whether to convert the emission file '
                         'created by sumo into a csv file')

if __name__ == "__main__":
    args = parser.parse_args()

    compute_action = get_compute_action_rllib(
        path_to_dir=args.result_dir,
        checkpoint_num=args.checkpoint_num,
        alg=args.run
    )

    # Collect the config data from the RLlib serialized files
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]
    config = get_rllib_config(result_dir)
    flow_params = get_flow_params(config)

    # Recreate the scenario from the pickled parameters
    exp_tag = flow_params["exp_tag"]
    net_params = flow_params['net']
    vehicles = flow_params['veh']
    initial_config = flow_params['initial']
    module = __import__("flow.scenarios", fromlist=[flow_params["scenario"]])
    scenario_class = getattr(module, flow_params["scenario"])
    module = __import__("flow.scenarios", fromlist=[flow_params["generator"]])
    generator_class = getattr(module, flow_params["generator"])

    scenario = scenario_class(name=exp_tag,
                              generator_class=generator_class,
                              vehicles=vehicles,
                              net_params=net_params,
                              initial_config=initial_config)

    # Start the environment with the gui turned on and a path for the
    # emission file
    module = __import__("flow.envs", fromlist=[flow_params["env_name"]])
    env_class = getattr(module, flow_params["env_name"])
    env_params = flow_params['env']
    sumo_params = flow_params['sumo']
    sumo_params.sumo_binary = "sumo-gui"
    sumo_params.emission_path = "./test_time_rollout/"

    env = env_class(env_params=env_params,
                    sumo_params=sumo_params,
                    scenario=scenario)

    # Run the environment in the presence of the pre-trained RL agent for the
    # requested number of time steps / rollouts
    exp = SumoExperiment(env=env, scenario=scenario)
    exp.run(num_runs=args.num_rollouts,
            num_steps=env.horizon,
            rl_actions=compute_action)

    # if prompted, convert the emission file into a csv file
    if args.emission_to_csv:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = "{0}-emission.xml".format(scenario.name)

        emission_path = \
            "{0}/test_time_rollout/{1}".format(dir_path, emission_filename)

        emission_to_csv(emission_path)
