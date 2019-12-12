'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-12-11 23:36:47
'''
import sys
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNBaseline
from heapq import nlargest
from collections import defaultdict

a = 0
top_n_baselineonly = ''
top_n_knnbasic = ''
top_n_knnbaseline = ''

for line in sys.stdin:
    line = line.strip()
    
    if a == 6:
        top_n_baselineonly = (line)
    elif a == 7:
        top_n_knnbasic = (line)
    elif a == 8:
        top_n_knnbaseline = (line)
    a += 1

def toDefaultDict(pred):
    n1 = defaultdict(list) 
    for key in pred:
        n1[key] = pred[key]
    return n1

top_n_baselineonly = toDefaultDict(eval(top_n_baselineonly))
top_n_knnbasic = toDefaultDict(eval(top_n_knnbasic))
top_n_knnbaseline = toDefaultDict(eval(top_n_knnbaseline))

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
    print("items_baselineonly == ",items_baselineonly)
    # return items_baselineonly
else:
    result = nlargest(5, rank, key=lambda s: rank[s])
    print("排名结果: {}".format(result))
    # return result