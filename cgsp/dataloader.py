"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import config
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Dataset(BasicDataset):

    def __init__(self, dataset, dtype, target=True, test_mode='tgt'):
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        
        dir = "data"
        if 'merge' in test_mode:
            test_mode = test_mode[-3:]
            data, self.m = self._merge(dataset, dtype, test_mode)
        else:
            domain = "tgt" if target else "src"
            data = pd.read_csv(f"{dir}/{dataset}_{dtype}_train_{domain}.csv")

        data = data[["uid", "iid"]]
        self.traindataSize = len(data)
        data = data.sample(frac=1)

        self.n_user = max(set(data.uid)) + 1
        self.m_item = max(set(data.iid)) + 1

        self.testDataSize = 0
        if target:
            print("extracting test dataset...")
            valData = pd.read_csv(f"{dir}/{dataset}_{dtype}_val_{test_mode}.csv")
            testData = pd.read_csv(f"{dir}/{dataset}_{dtype}_test_{test_mode}.csv")
            if test_mode == 'src':
                self.n_user = max(set(data.uid)) + 1
                self.m_item = max(set(data.iid)) + 1
                if dataset == 'douban' and test_mode == 'src':
                    self.n_user = 2719

            self.testDataSize = len(testData)
            print(f"{self.testDataSize} interactions for testing") 
            self.testDataSize = len(testData)
            self.testUniqueUsers = np.array(testData["uid"].unique())
            self.testUser = np.array(testData["uid"])
            self.testItem = np.array(testData["iid"])
        
        self.trainData = data
        self.trainUniqueUsers = np.array(data["uid"].unique())
        self.trainUser = np.array(data["uid"])
        self.trainItem = np.array(data["iid"])

        self.Graph = None
        print(f"{len(set(data.uid))} users and {len(set(data.iid))} items")
        print(f"{self.traindataSize} interactions for training")
        print(f"Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__valDict = None if not target else self.__build_test(np.array(valData["uid"]), np.array(valData["iid"]))
        self.__testDict = None if not target else self.__build_test(np.array(testData["uid"]), np.array(testData["iid"]))
        print(self.UserItemNet.shape)
        print(f"{dataset}-{dtype}-{target} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict
    
    @property
    def valDict(self):
        return self.__valDict

    @property
    def allPos(self):
        return self._allPos
    
        
    def _merge(self, dataset, dtype, test_mode):
        print("data merged")
        dir = ""
        src = pd.read_csv(f"{dir}/{dataset}_{dtype}_train_src.csv")
        tgt = pd.read_csv(f"{dir}/{dataset}_{dtype}_train_tgt.csv")
                
        print(f"source sparsity : {len(src)/max(src.uid)/max(src.iid)}")
        print(f"target sparsity : {len(tgt)/max(tgt.uid)/max(tgt.iid)}")
        print(f"overlap users : {len(set(src.uid) & set(tgt.uid))}")

        m = max(tgt.iid)
        src['iid'] = src['iid'].add(m+1)
        merge = pd.concat([tgt, src])
        return merge, m

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                # sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self, users, items):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(items):
            user = users[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            if user >= self.UserItemNet.shape[0]:
                posItems.append([])
            else:
                posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
