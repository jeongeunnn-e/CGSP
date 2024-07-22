import world
import utils
from world import cprint
import torch
import dataloader
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
# from register import dataset
from datetime import datetime
import json


def main():

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    try:

        src_data = dataloader.Dataset(dataset=world.dataset, dtype=world.dtype, target=False, test_mode=world.test_mode)
        des_data = dataloader.Dataset(dataset=world.dataset, dtype=world.dtype, target=True, test_mode=world.test_mode)

        if world.simple_model != 'none':
            epoch = 0
            cprint("[TEST]")
            print(f"[ model : {world.simple_model} alpha : {world.alpha} ]")
            results = Procedure.Test(src_data, des_data, f"{world.dataset}_{world.dtype}", world.alpha)
            with open(f'results/5-10/{world.dataset}_{world.dtype}_{world.simple_model}_{world.test_mode}_{datetime.now().strftime("%m%d%H%M%S")}.json', 'w') as f:
                json.dump(str(results), f)
                print(f'file saved')
        else: # lightGCN
            Recmodel = register.MODELS[world.model_name](world.config, des_data)
            Recmodel = Recmodel.to(world.device)
            bpr = utils.BPRLoss(Recmodel, world.config)
            start = time.time()
            for epoch in range(world.TRAIN_epochs):
                if epoch % 10 == 0:
                    cprint("[lightGCN]")
                    Procedure.Test_(des_data, Recmodel, epoch, w, world.config['multicore'])
                output_information = Procedure.BPR_train_original(des_data, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                torch.save(Recmodel.state_dict(), weight_file)
            end = time.time()
            print("****total time consumption : ", round(end-start,2))
    finally:
        if world.tensorboard:
            w.close()

if __name__ == "__main__":
    main()
