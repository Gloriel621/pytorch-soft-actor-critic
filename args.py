class Arguments:
    def __init__(self, 
    env_name = "InvertedPendulumBulletEnv-v0",
    eval_ = True,
    gamma = 0.99,
    tau = 0.005,
    lr = 0.0003,
    alpha = 0.2,
    automatic_entropy_tuning = False,
    seed = 1222621,
    batch_size = 256,
    num_steps = 100001,
    hidden_size = 256,
    updates_per_step = 1,
    start_steps = 10000,
    target_update_interval = 1,
    replay_size = 1000000,
    cuda = False
    ):
        self.env_name = env_name
        self.eval = eval_
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.seed = seed
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.target_update_interval = target_update_interval
        self.replay_size = replay_size
        self.cuda = cuda
