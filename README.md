# Transfering Hierarchical Structure with Dual Meta Imitation Learning

Code for Transfering Hierarchical Structure with Dual Meta Imitation Learning.

- [Transfering Hierarchical Structure with Dual Meta Imitation Learning](#transfering-hierarchical-structure-with-dual-meta-imitation-learning)
  - [Installation](#installation)
  - [Dataset](#dataset)
    - [Meta-world](#meta-world)
    - [Kitchen](#kitchen)
  - [General Usage](#general-usage)
    - [Meta-world](#meta-world-1)
    - [Kitchen](#kitchen-1)
  - [Detailed Instructions for Reproducibility](#detailed-instructions-for-reproducibility)
    - [Results of Table 1](#results-of-table-1)
    - [Results of Table 2](#results-of-table-2)
    - [Results of Table 3](#results-of-table-3)
    - [Results of Table 4](#results-of-table-4)
    - [Results of Table 5](#results-of-table-5)
    - [Results of Table 6](#results-of-table-6)

## Installation

1. Following [this](https://github.com/rlworkgroup/metaworld) to install Meta-world environments.
2. Following [this](https://github.com/rail-berkeley/d4rl) to install D4RL (for the Kitchen environment).
3. Clone this repository.

## Dataset

### Meta-world

Run `python ./metaworld/collect_demo.py` to get all demonstrations. By doing so, you will get a new folder `./metaworld/demos` that contains 2000 demonstrations for each of 50 environments in Meta-world suites.

### Kitchen

We use off-the-shelf demonstrations as mentioned on the page 15 of [FIST](https://openreview.net/pdf?id=xKZ4K0lTj_). There are a total of 24 multitask long horizon sets of trajectories that the data is collected from. Each trajectory set is sampled at least 10 times via VR tele-operation procedure.

However, preparing these demonstrations is extremely cumbersome. You need to follow [this](https://github.com/google-research/relay-policy-learning) to get and process original demonstration files. Firstly we need to transform original VR files into .pkl files, then we need to filter them to different tasks and store them as .npy files. Alternatively, we provide processed demonstrations here. You can go to `./kitchen/kitchen_demos` to get them and put all demos under a `kitchen_demos` folder under `./kitchen/` to run the code.

## General Usage

### Meta-world

You shall `cd ./metaworld` firstly. For training, run `python maml.py`. For testing, run `python evaluate.py` to get quantitative results and run `python evaluate_vis.py` to see robot manipulating in different environments. You should specify arguments when run above commands. Detailed information about arguments can be found in `arguments.py`.

### Kitchen

You shall `cd ./kitchen` firstly. For training, run `python maml.py`. For testing, run `python evaluate.py` to get quantitative results and run `python evaluate_vis.py` to see robot manipulating in the Kitchen environment. You should specify arguments when run above commands. Detailed information about arguments can be found in `arguments.py`.

## Detailed Instructions for Reproducibility

### Results of Table 1

`cd ./metaworld`.

Train: 

```python maml.py --sub_skill_cat=3 --continuous=True --suite='ML10'``` 

You could specify `--suite` to choose to train in `ML10` or `ML45`.

Test: 

```python evaluate.py --test_suite='ML10_test' --test_skill_num=3```

This will test the trained model on `ML10_test` suite and get success rates with 1-shot and 3-shot learning. You could specify `test_suite` to test in `ML10_train`, `ML10_test`, `ML45_train` and `ML45_test`. 


### Results of Table 2
`cd ./kitchen`.

Train: 

```python maml.py --sub_skill_cat=4 --continuous=True --suite='microwave_train'```. 

You could specify `--suite` to choose to train in `microwave_train`, `kettle_train`, `slider_train`, or `topburner_train`.

Test: 

```python evaluate.py --test_skill_num=4 --test_suite='microwave-kettle-topburner-light'```. 

This will test the trained model on `microwave-kettle-topburner-light` suite and get average cumulative rewards of 5 tries. You could specify `test_suite` to test in `microwave-kettle-topburner-light`, `microwave-bottomburner-light-slider`,  `microwave-kettle-hinge-slider`, or `microwave-kettle-hinge-slider`. Note you should match the training and testing suite to ensure the testing task is unseen task.

Results of __FIST-no-FT__ and __SPiRL__ are drawn from Table 2 in [FIST](https://openreview.net/pdf?id=xKZ4K0lTj_).

### Results of Table 3

`cd ./metaworld`

Specifying different sub_skill number __K__ during training and testing:

```
python maml.py --sub_skill_cat=K --continuous=True --suite='ML10'

python evaluate.py --test_skill_num=K --test_suite='microwave-kettle-topburner-light'
```

### Results of Table 4

`cd ./metaworld`

Specifying different variants during training:

```
python maml.py --sub_skill_cat=3 --continuous=True --suite='ML10' --dmil_high=True
python evaluate.py --test_suite='ML10_test' --test_skill_num=3 --dmil_high=True
```

or 

```
python maml.py --sub_skill_cat=3 --continuous=True --suite='ML10' --dmil_low=True
python evaluate.py --test_suite='ML10_test' --test_skill_num=3 --dmil_low=True
```


### Results of Table 5

Specifying different sub_skill number __K__ during training and testing:

Train: 

```
python maml.py --sub_skill_cat=K --continuous=True --suite='microwave_train'

python evaluate.py --test_skill_num=K --test_suite='microwave-kettle-topburner-light' 
```

Again you should match the training and testing suite to ensure the testing task is unseen task.


### Results of Table 6

Specify different finetune step N during testing:

```
python evaluate.py --test_skill_num=5  --test_suite='ML10_test' --finetune_step=N
```