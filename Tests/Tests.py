import subprocess
from pathlib import Path

from Analysis.Evaluation import load_and_evaluate_agent_from_file


def test_ppo():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestPPO --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config VanillaPPO --env_descr Ant-v4 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path('Test/TestPPO/run1_latest_test'), num_runs=1)

def test_ppo_no_norm():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestPPONoNorm --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config VanillaPPO --env_descr Ant-v4;norm:False --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestPPONoNorm/run1_latest_test"), num_runs=1)

def test_cascade():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCascade --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config Cascade --env_descr Ant-v4 --agent_params base_steps:300 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCascade/run1_latest_test"), num_runs=1)

def test_cascade_propagation():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCascadePropagation --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config Cascade --env_descr Ant-v4 --agent_params base_steps:300;propagate_value:True;propagate_action:True --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCascadePropagation/run1_latest_test"), num_runs=1)

def test_cascade_not_sequential():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCascadeNotSequential --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config Cascade --env_descr Ant-v4 --agent_params base_steps:500;sequential:False;stacks:3;propagate_action:True --save_latest",
        shell=True,
        check=True,
)
    load_and_evaluate_agent_from_file(Path("Test/TestCascadeNotSequential/run1_latest_test"), num_runs=1)

def test_cascade_naive():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCascadeNaive --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config CascadeNaive --env_descr Ant-v4 --agent_params base_steps:300;propagate_value:True;propagate_action:True --save_latest",
        shell=True,
        check=True,
)
    load_and_evaluate_agent_from_file(Path("Test/TestCascadeNaive/run1_latest_test"), num_runs=1)
