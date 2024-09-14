# OARL
This repository is the implementation of our research "**[Robust Lane Change Decision Making for Autonomous Vehicles: An Observation Adversarial Reinforcement Learning Approach](https://www.researchgate.net/publication/359776714_Robust_Lane_Change_Decision_Making_for_Autonomous_Vehicles_An_Observation_Adversarial_Reinforcement_Learning_Approach)**". This study has been published in [IEEE TIV](https://ieeexplore.ieee.org/abstract/document/9750867). 

## Introduction
### Schematic of the OARL Framework toward Robust Lane Change Decision Making for Autonomous Vehicles
<img src="/framework.jpg" alt="ENV" width="600" height="500">
Reinforcement learning holds the promise of allowing autonomous vehicles to learn complex decision making behaviors through interacting with other traffic participants.
However, many real-world driving tasks involve unpredictable perception errors or measurement noises which may 
mislead an autonomous vehicle into making unsafe decisions, even cause catastrophic failures.
In light of these risks, to ensure safety under perception uncertainty, autonomous vehicles are required to be able to cope with the worst case observation perturbations.
Therefore, this paper proposes a novel observation adversarial reinforcement learning approach for robust lane change decision making of autonomous vehicles.
A constrained observation-robust Markov decision process is presented to model lane change decision making behaviors of autonomous vehicles under policy constraints and observation uncertainties. 
Meanwhile, a black-box attack technique based on Bayesian optimization is implemented to approximate the optimal adversarial observation perturbations efficiently.
Furthermore, a constrained observation-robust actor-critic algorithm is advanced to optimize autonomous driving lane change policies while keeping the variations of the policies attacked by the optimal adversarial observation perturbations within bounds.
Finally, the robust lane change decision making approach is evaluated in three stochastic mixed traffic flows based on different densities.
The results demonstrate that the proposed method can not only enhance the performance of an autonomous vehicle but also improve the robustness of lane change policies against adversarial observation perturbations.

## User Guidance
### Installation
This repo is developed using Python 3.7 and PyTorch 1.3.1+CPU in Ubuntu 16.04. 

We utilize the proposed FNI-RL approach to train the autonomous driving agent in the popular [Simulation of Urban Mobility](https://eclipse.dev/sumo/) (SUMO, Version 1.2.0) platform.

We believe that our code can also run on other operating systems with different versions of Python, PyTorch and SUMO, but we have not verified it.

The required packages can be installed using

	pip install -r requirements.txt

### Run
 Users can leverage the following command to run the code in the terminal and train the autonomous driving agent in highway scenarios with stochastic mixed trafﬁc ﬂows:

	python main.py



## Acknowledgement
We greatly appreciate the important references provided by the two code repositories [BO](https://github.com/bayesian-optimization/BayesianOptimization) and [SAC](https://github.com/denisyarats/pytorch_sac) for the implementation of our research.

## Citation
If you find this repository helpful for your research, we would greatly appreciate it if you could star our repository and cite our work.
```
@ARTICLE{9750867,
  author={He, Xiangkun and Yang, Haohan and Hu, Zhongxu and Lv, Chen},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Robust Lane Change Decision Making for Autonomous Vehicles: An Observation Adversarial Reinforcement Learning Approach}, 
  year={2023},
  volume={8},
  number={1},
  pages={184-193},
  keywords={Decision making;Autonomous vehicles;Perturbation methods;Optimization;Bayes methods;Task analysis;Safety;Adversarial attack;autonomous vehicle;lane change decision making;reinforcement learning;robust decision making},
  doi={10.1109/TIV.2022.3165178}}
```
