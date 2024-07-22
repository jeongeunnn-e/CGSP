"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
import time
from dataloader import BasicDataset
from torch import nn
import scipy.sparse as sp
import numpy as np
# from sparsesvd import sparsesvd

import dataloader
from build_matrix import build as build_mat

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
class LGCN_IDE(object):
    def __init__(self, data : BasicDataset):
        self.adj_mat = data.UserItemNet.tolil()
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        end = time.time()
        print('training time for LGCN-IDE', end-start)
        
    def getUsersRating(self, batch_users, ds_name='douban'):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users,:].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj # @ norm_adj.T @ norm_adj
        return U_1
        
class GF_CF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sp.linalg.svds(self.norm_adj, 256)
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, batch_users):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if world.svd:
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + 0.3 * U_1
        ret = U_2
        return ret
    

class PGSP(object):
    def __init__(self, data : BasicDataset):
        self.adj_mat = data.UserItemNet.tolil()
        self.n_users = data.n_users
        self.linear=True
    
    def build_aug_sm_graph(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten() 
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat) #normalized R = S_ui        s
        norm_adj = norm_adj.tocsc() #normalized R = S_ui

        s_ui = norm_adj
        s_uu = s_ui @ s_ui.T
        s_ii = s_ui.T @ s_ui
        
        aug_above = sp.hstack([s_uu, s_ui], dtype=np.float32)
        self.norm_r = sp.hstack([s_uu, s_ui], dtype=np.float32)
        aug_below = sp.hstack([s_ui.T, s_ii], dtype=np.float32)
        aug_sm = sp.vstack([aug_above, aug_below])

        return aug_sm
    
    def train(self, linear=True):
        start = time.time()
        self.aug = self.build_aug_sm_graph(self.adj_mat)
        if world.svd:
            self.linear = False
            adj_mat = self.aug
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(adj_mat)
            colsum = np.array(adj_mat.sum(axis=0))
            d_inv = np.power(colsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            self.d_mat_i = d_mat
            self.d_mat_i_inv = sp.diags(1/d_inv)
            norm_adj = norm_adj.dot(d_mat)
            ut, s, self.vt = sp.linalg.svds(norm_adj, 256)
        end = time.time()
        print('training time for PGSP', end-start)
      
    def getUsersRating(self, batch_users):
        norm_adj = np.array(self.norm_r.todense())
        adj_mat = self.adj_mat
        aug = np.array(self.aug.todense())
        R = sp.hstack([norm_adj, adj_mat], dtype=np.float32).todense()
        batch_test = R[batch_users,:]
        if self.linear:
            ret = batch_test @ aug
        else:
            U_2 = batch_test @ aug
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = 0.7*U_2 + 0.3 * U_1
        ret = ret[:, self.n_users:]
        return ret
    

class CGSP_IO(object):
    def __init__(self, src_data : BasicDataset, des_data : BasicDataset, o_users, test_mode):
        self.src_adj_mat = src_data.UserItemNet.tolil()
        self.des_adj_mat = des_data.UserItemNet.tolil()
        self.m_items = des_data.m_items
        self.n_users = des_data.n_users
        self.o_users = o_users
        self.test_mode= test_mode

    def ideal_low_pass(self, sm):
        rowsum = np.array(sm.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(sm)

        colsum = np.array(sm.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        d_mat_i_inv = sp.diags(1/d_inv)
        ut, s, self.vt = sp.linalg.svds(norm_adj, 256)
        self.U2 = d_mat_i @ self.vt.T @ self.vt @ d_mat_i_inv
        return

    def train(self, alpha):
        r_s = self.normalize(self.src_adj_mat)
        r_t = self.normalize(self.des_adj_mat)
        r_os = r_s[self.o_users,:] 
        r_ot = r_t[self.o_users,:] 
        
        ret = r_ot.T @ r_os @ r_os.T @ r_ot @ r_t.T @ r_t 
        mat = r_t.T @ r_t
        self.item_sm = alpha * ret + (1-alpha) * mat
            
        if world.svd:
            self.ideal_low_pass(self.item_sm)

    def normalize(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        return norm_adj 
    
    def getUsersRating(self, batch_users):
        
        # build input signal
        if self.test_mode == 'tgt':
            R = self.normalize(self.des_adj_mat)
        elif self.test_mode == 'src':
            r_s = self.normalize(self.src_adj_mat)
            r_t = self.normalize(self.des_adj_mat)
            r_os = r_s[self.o_users,:] 
            r_ot = r_t[self.o_users,:]
            R = r_s @ r_os.T @ r_ot
        
        item_sm = self.item_sm
        batch_test = np.array(R[batch_users,:].todense())
        ret = batch_test @ item_sm
        return ret


class CGSP_OA(GF_CF):
    def __init__(self, src_data : BasicDataset, des_data : BasicDataset, o_users, test_mode):
        self.src_adj_mat = src_data.UserItemNet.tolil()
        self.des_adj_mat = des_data.UserItemNet.tolil()
        self.n_users = des_data.n_users
        self.n_users_src = src_data.n_users
        self.m_items = des_data.m_items
        self.o_users = o_users
        self.test_mode = test_mode
    
    def ideal_low_pass_(self, sm):
        rowsum = np.array(sm.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(sm)

        colsum = np.array(sm.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = np.diag(d_inv)
        d_mat_i = d_mat
        d_mat_i_inv = np.diag(1/d_inv)
        ut, s, self.vt = np.linalg.svd(norm_adj, 256)
        U2 = d_mat_i @ self.vt.T @ self.vt @ d_mat_i_inv
        return U2

    def normalize(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        return norm_adj 
    
    def train(self, alpha=0.6):
        
        r_s = self.normalize(self.src_adj_mat)
        r_t = self.normalize(self.des_adj_mat)
        r_os = r_s[self.o_users,:] 
        r_ot = r_t[self.o_users,:] 

        ret = r_ot @ r_ot.T @ r_os @ r_os.T @ r_ot @ r_ot.T
        mat = r_ot @ r_ot.T
        s_u = alpha * ret + (1-alpha) * mat

        ret = r_os @ r_s.T @ r_s @ r_os.T @ r_ot @ r_t.T @ r_t
        mat = r_ot
        s_ui = alpha * ret + (1-alpha) * mat
        aug_above = sp.hstack([ s_u, s_ui ], dtype=np.float32)

        ret = r_ot.T @ r_os @ r_os.T @ r_ot @ r_t.T @ r_t 
        mat = r_t.T @ r_t
        s_i = alpha * ret + (1-alpha) * mat
        aug_below = sp.hstack([ s_ui.T , s_i ], dtype=np.float32)
        aug_sm = sp.vstack([ aug_above, aug_below ], dtype=np.float32)
        self.aug_sm = np.array(aug_sm.todense())
        return

    def getUsersRating(self, batch_users):

        r_s = self.normalize(self.src_adj_mat)
        r_t = self.normalize(self.des_adj_mat)
        r_os = r_s[self.o_users,:] 
        r_ot = r_t[self.o_users,:]

        if self.test_mode == 'tgt':
            s_u = r_t @ r_ot.T
            R = sp.hstack([s_u, r_t], dtype=np.float32).todense()
        elif self.test_mode == 'src':
            s_u = r_s @ r_os.T
            s_ui = s_u  @ r_ot
            R = sp.hstack([s_u, s_ui], dtype=np.float32).todense()
        self.R = R
        
        R = torch.from_numpy(self.R).float()
        aug_sm = torch.from_numpy(self.aug_sm)
        U = R[batch_users, :] @ aug_sm
        U = U.cpu().numpy()
        m = self.m_items
        return U[:, -m:]

class CGSP_UA(GF_CF):
    def __init__(self, src_data : BasicDataset, des_data : BasicDataset, o_users, test_mode):
        self.src_adj_mat = src_data.UserItemNet.tolil()
        self.des_adj_mat = des_data.UserItemNet.tolil()
        self.n_users = des_data.n_users
        self.m_items = des_data.m_items
        self.o_users = o_users
        self.test_mode = test_mode
    
    def ideal_low_pass_(self, sm):
        rowsum = np.array(sm.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(sm)

        colsum = np.array(sm.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = np.diag(d_inv)
        d_mat_i = d_mat
        d_mat_i_inv = np.diag(1/d_inv)
        ut, s, self.vt = np.linalg.svd(norm_adj, 256)
        U2 = d_mat_i @ self.vt.T @ self.vt @ d_mat_i_inv
        return U2

    def normalize(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        return norm_adj 
    
    def train(self, alpha=0.):
        
        r_s = self.normalize(self.src_adj_mat)
        r_t = self.normalize(self.des_adj_mat)
        r_os = r_s[self.o_users,:] 
        r_ot = r_t[self.o_users,:] 

        ret = r_t @ r_ot.T @ r_os @ r_os.T @ r_ot @ r_t.T
        mat = r_t @ r_t.T
        s_u = alpha * ret + (1-alpha) * mat

        ret = r_t @ r_ot.T @ r_os @ r_s.T @ r_s @ r_os.T @ r_ot @ r_t.T @ r_t
        mat = r_t
        s_ui = alpha * ret + (1-alpha) * mat

        aug_above = sp.hstack([s_u, s_ui], dtype=np.float32)

        ret = r_ot.T @ r_os @ r_os.T @ r_ot @ r_t.T @ r_t 
        mat = r_t.T @ r_t
        s_i = alpha * ret + (1-alpha) * mat
        aug_below = sp.hstack([ s_ui.T, s_i ], dtype=np.float32)
        aug_sm = sp.vstack([ aug_above, aug_below ], dtype=np.float32)
        self.aug_sm = np.array(aug_sm).todense()
        return

    def getUsersRating(self, batch_users):

        r_s = self.normalize(self.src_adj_mat)
        r_t = self.normalize(self.des_adj_mat)
        r_os = r_s[self.o_users,:] 
        r_ot = r_t[self.o_users,:] 

        if self.test_mode == 'tgt':
            s_u = r_t @ r_t.T
            R = sp.hstack([s_u, r_t], dtype=np.float32).todense()
        elif self.test_mode == 'src':
            s_ui = r_s @ r_os.T @ r_ot
            s_u = s_ui @ r_t.T
            R = sp.hstack([s_u, s_ui], dtype=np.float32).todense()

        R = torch.from_numpy(self.R).float()
        aug_sm = torch.from_numpy(self.aug_sm).float()
        U = R[batch_users, :] @ aug_sm
        m = self.m_items
        return U[:, -m:]
