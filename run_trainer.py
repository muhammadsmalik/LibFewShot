# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer

    
            
if __name__ == "__main__":

    name="CAN_Conv64F_1"


    # reset final results
    f = open("final_result.csv", "w")
    f.write("")
    f.close()
    
    f = open("final_result.csv", "a")

    f.write("Model" + "," + "Number of Shots" + "," + "Backbone" + ","+"Train Accuracy" + "," + "Best Train Accuracy" + "," + "Test 1 Accuracy" + ","+"Test 1 Best Accuracy" + ","+"Validation Accuracy"+"," + "Best Validation Accuracy" +","+"Test 2 Final Accuracy" + "," + "Test 2 Best Accuracy\n")
        
    f.write(name + "," + str(1) + "," + "Conv64F" )


    config = Config("config/test_install.yaml").get_config_dict()
    rank = 0  # Set the rank to 0 for single GPU or CPU
    trainer = Trainer(rank, config)  # Pass both rank and config arguments
    # trainer = Trainer(rank, config, name, f, printName) 
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