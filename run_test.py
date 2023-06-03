# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./results/ProtoNet-consolidated_seeds_dataset-Conv64F-2-1-Jun-03-2023-15-20-14"
VAR_DICT = {
    "test_epoch": 5,
    "n_gpu": 0,
    "test_episode": 600,
    "episode_size": 1,
}
    


if __name__ == "__main__":
    print("Entered main")
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

    # if config["n_gpu"] > 1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    #     torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    # else:
    print("Before test loop: Init test")
    test = Test(0, config, PATH)
    print("Before test loop: Start")
    test.test_loop()
