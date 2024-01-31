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

def test_chooser_env():
    test_ppo() #First run PPO to get the agents for chooser env

    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestChooserEnv --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config DiscreteChooser --env_descr ChooserEnv;env:Ant-v4,norm=False,agent_config=VanillaPPO;agents:[Test/TestPPO/run1_latest_test,Test/TestPPO/run1_latest_test,Test/TestPPO/run1_latest_test,Test/TestPPO/run1_latest_test] --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestChooserEnv/run1_latest_test"), num_runs=1)

def test_ppo_no_norm():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestPPONoNorm --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config VanillaPPO --env_descr Ant-v4;norm:False --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestPPONoNorm/run1_latest_test"), num_runs=1)

def test_discrete_chooser():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestDiscreteChooser --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config DiscreteChooser --env_descr Acrobot-v1 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestDiscreteChooser/run1_latest_test"), num_runs=1)


def test_combn():
    subprocess.run(
        "python ../mainExperiment.py   --exp_group Test --exp_name TestCombN --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config CombN --env_descr Ant-v4 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCombN/run1_latest_test"), num_runs=1)

def test_combn_entropies():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCombNEntropies --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config CombN --env_descr Ant-v4 --agent_params hent_coef:-0.1;vent_coef:0.1 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCombNEntropies/run1_latest_test"), num_runs=1)


def test_combn_alternating():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCombNAlternating --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config CombN --env_descr Ant-v4 --agent_params chooser_epochs:1;action_epochs:1 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCombNAlternating/run1_latest_test"), num_runs=1)


def test_combn_hierarchical():
    subprocess.run(
        "python ../mainExperiment.py   --exp_group Test --exp_name TestCombNHierarchical --exp_identifier test --steps 700 --eval_interval 700 --agent_config CombNHierarchical --env_descr Ant-v4 --agent_params n:2;cycles:2;cycle_steps:100;vent:0.1 --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCombNHierarchical/run1_latest_test"), num_runs=1)


def test_combn_hierarchical_no_pretrain():
    subprocess.run(
        "python ../mainExperiment.py  --exp_group Test --exp_name TestCombNHierarchicalNoPretrain --exp_identifier test --steps 1000 --eval_interval 1000 --agent_config CombNHierarchical --env_descr Ant-v4 --agent_params n:2;cycles:2;cycle_steps:100;vent:0.1;pretrain:False --save_latest",
        shell=True,
        check=True,
    )
    load_and_evaluate_agent_from_file(Path("Test/TestCombNHierarchicalNoPretrain/run1_latest_test"), num_runs=1)


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
