
import sys
import time
sys.dont_write_bytecode = True
from core.config import Config
from core import Trainer

import os
import torch
from core import Test

if __name__ == "__main__":
    

    

    nameCollection=["SKD_5_Conv64F_1","SKD_5_resnet12_1","SKD_5_resnet18_1"]

    modelCollection=["SKD","SKD","SKD"]

    numShotsCollection=["5","5","5"]

    backboneCollection=["Conv64F","resnet12","resnet18"]

    # reset final results
    f = open("final_result.csv", "w")
    f.write("")
    f.close()
    
    f = open("final_result.csv", "a")

    f.write("Model" + "," + "Number of Shots" + "," + "Backbone" + ","+"Train Accuracy" + "," + "Best Train Accuracy" + "," + "Test 1 Accuracy" + ","+"Test 1 Best Accuracy" + ","+"Validation Accuracy"+"," + "Best Validation Accuracy" +","+"Test 2 Final Accuracy" + "," + "Test 2 Best Accuracy\n")
    total_count = len(nameCollection)
    total_count_index = 0
    for name in nameCollection:
        
        printName = "Code is done with: " + name + " with progress: "+ str(total_count_index) + "/" + str(total_count)
        
        
        # name = "test_run"

        fileName = name + ".yaml"
        
        f.write(modelCollection[total_count_index] + "," + str(numShotsCollection[total_count_index]) + "," + backboneCollection[total_count_index] )

        config = Config("config/"+fileName).get_config_dict()
        rank = 0  # Set the rank to 0 for single GPU or CPU
        trainer = Trainer(rank, config, name, f,printName)  # Pass both rank and config arguments
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

        test = Test(0, config, f, printName, PATH)
        
        test.test_loop()

        f.write("\n")

        os.rename("data/consolidated_seeds_dataset/test.csv", "data/consolidated_seeds_dataset/test_2.csv")
        time.sleep(1)
        os.rename("data/consolidated_seeds_dataset/test_temp.csv", "data/consolidated_seeds_dataset/test.csv")
        

        # update progress bar
        # update progress bar
        total_count_index = total_count_index + 1
                    
                    

    f.close()
