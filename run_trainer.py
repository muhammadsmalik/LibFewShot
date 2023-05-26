# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer

if __name__ == "__main__":
    config = Config("/Users/salmanmalik/Documents/MPhil/LibFewShot/config/test_install.yaml").get_config_dict()
    rank = 0  # Set the rank to 0 for single GPU or CPU
    trainer = Trainer(rank, config)  # Pass both rank and config arguments
    trainer.train_loop(rank)

# # -*- coding: utf-8 -*-
# import sys

# sys.dont_write_bytecode = True

# import torch
# import os
# from core.config import Config
# from core import Trainer


# def main(rank, config):
#     trainer = Trainer(rank, config)
#     trainer.train_loop(rank)


# if __name__ == "__main__":
#     config = Config("./config/proto.yaml").get_config_dict()

#     if config["n_gpu"] > 1:
#         os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
#         torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
#     else:
#         main(0, config)