'''
@Description: This is a python file about collaborative filtering
@Author: Group 6
@Date: 2019-11-09 09:43:09
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

import time
import sys
import os
import pymysql
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNBaseline
from heapq import nlargest


'''
@description: 
@Author: zequn
'''
def get_top_n(pred, n=10):
    # 得到前n个value最大对key
    n1 = defaultdict(list) #如果没有则返回[]
    #对于每个user分别采集
    for u,i,e, _ in pred:
        n1[u].append((i, e))
    # 每一个用户取最高对几个item
    for key, val in n1.items():
        n1[u] = nlargest(n, val, key=lambda s: s[1])
    return n1



def user_build_anti_testset(train, uid, f=None):
    # 为用户生成测试集
    # 
    f = train.global_mean if f is None else float(f)

    i_id = train.to_inner_uid(uid)
    test = []

    item = set([j for (j, _) in train.ur[i_id]])
    test += [(uid, train.to_raw_iid(i), f) for
                     i in train.all_items() if
                     i not in item]
    return test

'''
@description: this is a class about KNN
@Author: JeanneWu
'''
class PredictionSet():

    '''
    @description: 寻找k个临近的点
    @param user_raw_id 用户的id
    @param trainset 训练集
    @return: 
    '''
    def __init__(self, algorithm, trainset, raw_user_id=None, k=40):
        # 先把传入的参数定义到类里方便后面调用
        self.algorithm = algorithm
        self.trainset = trainset
        self.k = k
        if raw_user_id is not None:
            self.r_uid = raw_user_id
            self.i_uid = trainset.to_inner_uid(raw_user_id)
            self.knn_userset = self.algorithm.get_neighbors(self.i_uid, self.k) # 得到K个最相似的用户
            print(self.knn_userset,'knn_userset')
            
            #把j去重后放到user_items里面
       
            user_items = self.generate_user_item()
       
            self.neighbor_items = set()
            for nnu in self.knn_userset:
                for (j, _) in trainset.ur[nnu]:
                    if j not in user_items:
                        self.neighbor_items.add(j)

    def user_build_anti_testset(self, fill=None):
        """
            为单个用户生成测试集
        """
        if fill is None:
            fill = self.trainset.global_mean
        else:
            float(fill)
        
        unique_testset = []
        user_items = self.generate_user_item()
        unique_testset += [(self.r_uid, self.trainset.to_raw_iid(i), fill) for
                         i in self.neighbor_items if
                         i not in user_items]
        return unique_testset

    def generate_user_item(self):
        #把j去重后放到user_items里面
        temArr = []
        for (j, _) in self.trainset.ur[self.i_uid]:
            temArr.append(j)
        user_items_func = set(temArr)
        return user_items_func

'''
@description: this is a function about collaborative_filtering
@Author: SuSu
'''
def collaborative_filtering(raw_uid):
    # To read the data from a txt file
    # TODO: To modify the file path of the data set
    file_path = ""

    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)

    # Build the training set
    trainset = data.build_full_trainset()

    # Baselineonly
    bsl_options = {'method': 'sgd',
                    'learning_rate': 0.0005,
                 }
    algo_BaselineOnly = BaselineOnly(bsl_options=bsl_options)
    algo_BaselineOnly.fit(trainset)

    rset = user_build_anti_testset(trainset, raw_uid)
    predictions = algo_BaselineOnly.test(rset)
    top_n_baselineonly = get_top_n(predictions, n=5)

    # KNNBasic
    sim_options = {'name': 'pearson', 'user_based': True}
    algo_KNNBasic = KNNBasic(sim_options=sim_options)
    algo_KNNBasic.fit(trainset)

    predictor = PredictionSet(algo_KNNBasic, trainset, raw_uid)
    knn_anti_set = predictor.user_build_anti_testset()
    predictions = algo_KNNBasic.test(knn_anti_set)
    top_n_knnbasic = get_top_n(predictions, n=5)

    # KNNBaseline
    sim_options = {'name': 'pearson_baseline', 'user_based': True}
    algo_KNNBaseline = KNNBaseline(sim_options=sim_options)
    algo_KNNBaseline.fit(trainset)

    predictor = PredictionSet(algo_KNNBaseline, trainset, raw_uid)
    knn_anti_set = predictor.user_build_anti_testset()
    predictions = algo_KNNBaseline.test(knn_anti_set)
    top_n_knnbaseline = get_top_n(predictions, n=5)

    recommendset = set()
    for results in [top_n_baselineonly, top_n_knnbasic, top_n_knnbaseline]:
        for key in results.keys():
            for recommendations in results[key]:
                iid, rating = recommendations
                recommendset.add(iid)

    items_baselineonly = set()
    for key in top_n_baselineonly.keys():
        for recommendations in top_n_baselineonly[key]:
            iid, rating = recommendations
            items_baselineonly.add(iid)

    items_knnbasic = set()
    for key in top_n_knnbasic.keys():
        for recommendations in top_n_knnbasic[key]:
            iid, rating = recommendations
            items_knnbasic.add(iid)

    items_knnbaseline = set()
    for key in top_n_knnbaseline.keys():
        for recommendations in top_n_knnbaseline[key]:
            iid, rating = recommendations
            items_knnbaseline.add(iid)

    rank = dict()
    for recommendation in recommendset:
        if recommendation not in rank:
            rank[recommendation] = 0
        if recommendation in items_baselineonly:
            rank[recommendation] += 1
        if recommendation in items_knnbasic:
            rank[recommendation] += 1
        if recommendation in items_knnbaseline:
            rank[recommendation] += 1

    max_rank = max(rank, key=lambda s: rank[s])
    if max_rank == 1:
        return items_baselineonly
    else:
        result = nlargest(5, rank, key=lambda s: rank[s])
        return result
    

    