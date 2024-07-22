'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime
import time
import math


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
#    print(sorted_items[0], groundTrue[0])
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def evaluate(multicore, u_batch_size, lm, des_data, max_K, m, test):
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    testDict = des_data.testDict if test else des_data.valDict
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        # print(total_batch)
        idx = 0
        for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size)):
            # print(f"working on batch {idx}")
            idx = idx+1
            allPos = des_data.getUserPosItems(batch_users) # if target == 'coldstart' else []
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            rating = lm.getUsersRating(batch_users)
            rating = torch.from_numpy(rating)
            if test:
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K) if m <0 else torch.topk(rating[:,:m+1], k=max_K)
            rating = rating.cpu().numpy()
            del rating
            # print(total_batch, len(users_list), batch_users)
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            # w.add_scalars
            print(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))})
            # w.add_scalars
            print(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))})
            # w.add_scalars
            print(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))})
        if multicore == 1:
            pool.close()
        print(results)
        return results
                   
def Test(src_data, des_data, ds_name, alpha=0, w=None, multicore=0):
    u_batch_size = 100 if world.dataset == 'douban' else 264 
    src_data: utils.BasicDataset
    des_data: utils.BasicDataset
    
    #overlapping user 
    src_users = src_data.trainUniqueUsers
    des_users = des_data.trainUniqueUsers
    o_users = list(set(src_users) & set(des_users))
    print(f"overlapping users are {len(o_users)}")

    start = time.time()
    if 'cgsp-io'==world.simple_model:
        lm = model.CGSP_IO(src_data, des_data, o_users, world.test_mode)
        lm.train(alpha)
    elif 'cgsp-oa'==world.simple_model: 
        lm = model.CGSP_OA(src_data, des_data, o_users, world.test_mode)
        lm.train(alpha)
    elif 'cgsp-ua'==world.simple_model: 
        lm = model.CGSP_OA(src_data, des_data, o_users, world.test_mode)
        lm.train(alpha)

    # BASELINES
    elif 'gf-cf'==world.simple_model:
        lm = model.GF_CF(des_data.UserItemNet.tolil())
        lm.train() 
    elif 'pgsp'==world.simple_model: 
        lm = model.PGSP(des_data, world.svd)
        lm.train()
    elif world.simple_model == 'lgcn-ide':
        lm = model.LGCN_IDE(des_data)
        lm.train()
    end = time.time()
    print("***total training time ", end-start)
    max_K = max(world.topks)
    
    m = des_data.m if 'merge' in world.test_mode else -1
    val= evaluate(multicore, u_batch_size, lm, des_data, max_K, m, test=False)
    test = evaluate(multicore, u_batch_size, lm, des_data, max_K, m, test=True)
    return val, test
