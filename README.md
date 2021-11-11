### Description
------------
Reimplementation of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) and a deterministic variant of SAC from [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

### Changes in Forked version

Changed mujoco environments to pybullet since I found it bothering to install mujoco.
Deleted command line options.(Change arguments manually)
Deleted tensorboard options.
Deleted Deterministic Policy model.

### Requirements
------------
*   [pybullet](https://pybullet.org/)
*   [PyTorch](http://pytorch.org/)

### Default Arguments and Usage
------------
### Usage

```
usage: python main.py 
```


### Arguments
------------
```
set args in main.py.

  policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  eval EVAL           Evaluates a policy a policy every 10 episode (default:
                        True)
  gamma G             discount factor for reward (default: 0.99)
  tau G               target smoothing coefficient(τ) (default: 5e-3)
  lr G                learning rate (default: 3e-4)
  alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  automatic_entropy_tuning G
                        Automaically adjust α (default: False)
  seed N              random seed (default: 123456)
  batch_size N        batch size (default: 256)
  num_steps N         maximum number of steps (default: 1e6)
  hidden_size N       hidden size (default: 256)
  updates_per_step N  model updates per simulator step (default: 1)
  start_steps N       Steps sampling random actions (default: 1e4)
  target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  replay_size N       size of replay buffer (default: 1e6)
  cuda                run on CUDA (default: False)
```

| Environment **(`--env-name`)**|
| ---------------| -------------|
| InvertedPendulumBulletEnv-v0|
