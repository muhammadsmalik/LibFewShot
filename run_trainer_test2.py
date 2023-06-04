# -*- coding: utf-8 -*-
import sys
import time
sys.dont_write_bytecode = True
from alive_progress import alive_bar
from core.config import Config
from core import Trainer

import os
import torch
from core import Test

if __name__ == "__main__":
    
    modelCollection = ["Proto", 
                   "RelationNet", 
                   "ConvMNet", 
                   "DN4", 
                   "CAN", 
                   "ATL_NET", 
                   "ADM", 
                   "FEAT", 
                   "RENet", 
                   "DeepBdc"]
    numberOfShotsCollection = [1,5,10]
    backbonesCollection = ["Conv64F", "resnet12", "resnet18", "Conv32F"]
    trialRunCollection = [1,2,3]

    # reset final results
    f = open("final_result.csv", "w")
    f.write("")
    f.close()
    
    f = open("final_result.csv", "a")

    f.write("Model" + "," + "Number of Shots" + "," + "Backbone" + ","+"Trial Number"+","+"Train Accuracy" + "," + "Best Train Accuracy" + "," + "Test 1 Accuracy" + ","+"Test 1 Best Accuracy" + ","+"Validation Accuracy"+"," + "Best Validation Accuracy" +","+"Test 2 Final Accuracy" + "," + "Test 2 Best Accuracy\n")
    total_count = len(numberOfShotsCollection)*len(modelCollection)*len(backbonesCollection)*len(trialRunCollection)
    with alive_bar(total_count) as  bar:
        for numShots in numberOfShotsCollection:
            for model in modelCollection:
                for backbone in backbonesCollection:
                    for trial in trialRunCollection:
                        name = model+"_"+str(numShots)+"_"+backbone+"_"+str(trial)
                        # name = "test_run"

                        fileName = name + ".yaml"
                        
                        f.write(model + "," + str(numShots) + "," + backbone + ","+str(trial)+",")

                        config = Config("config/"+fileName).get_config_dict()
                        rank = 0  # Set the rank to 0 for single GPU or CPU
                        trainer = Trainer(rank, config, name, f)  # Pass both rank and config arguments
                        trainer.train_loop(rank)

                        
                        # swap out the test for a different test folder
                        os.rename("data/consolidated_seeds_dataset/test.csv", "data/consolidated_seeds_dataset/test_temp.csv")
                        time.sleep(1)
                        os.rename("data/consolidated_seeds_dataset/test_2.csv", "data/consolidated_seeds_dataset/test.csv")

                        PATH = "./results/"+name
                        VAR_DICT = {
                            "test_epoch": 5,
                            "n_gpu": 1,
                            "test_episode": 100,
                            "episode_size": 1,
                        }

                        config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

                        test = Test(0, config, f, PATH)
                        
                        test.test_loop()

                        f.write("\n")

                        os.rename("data/consolidated_seeds_dataset/test.csv", "data/consolidated_seeds_dataset/test_2.csv")
                        time.sleep(1)
                        os.rename("data/consolidated_seeds_dataset/test_temp.csv", "data/consolidated_seeds_dataset/test.csv")
                        

                        # update progress bar
                        bar.text("Completed: " + name)
                    

    f.close()

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