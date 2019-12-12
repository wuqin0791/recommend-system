from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

import time
import sys
import os
import pymysql
import numpy as np
import pandas as pd
import io
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNBaseline
from heapq import nlargest
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

class ItemBasedRecommender():
    def __init__(self):
        self.train_data = None
        self.uid = None
        self.iid = None
        self.cooccurence = None
        self.songs = None
        self.rev_songs = None
        self.item_similarity_recommendations = None

    def read_data(self, train_data, uid, iid):
        self.train_data = train_data
        self.uid = uid
        self.iid = iid

    def get_user_songs(self, user):
        user_data = self.train_data[self.train_data[self.uid] == user]
        user_songs = list(user_data[self.iid].unique())

        return user_songs
    
    def get_song_users(self, song):
        song_data = self.train_data[self.train_data[self.iid] == song]
        song_users = set(song_data[self.uid].unique())
            
        return song_users

    def get_all(self):
        return list(self.train_data[self.iid].unique())
    
    def get_all_song_name(self):
        return list(self.train_data["song_name"].unique())

    def set_coocurrence(self, user_songs, all_songs):
        users = []
        
        for song in user_songs:
            users.append(self.get_song_users(song))
    
        co_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(len(all_songs)):
            songs_i_data = self.train_data[self.train_data[self.iid] == all_songs[i]]
            user_set1 = set(songs_i_data[self.uid].unique())

            for j in range(len(user_songs)):
                user_set2 = users[j]
                user_intersect = user_set1.intersection(user_set2)
                if len(user_intersect) != 0:
                    user_union = user_set1.union(user_set2)
                    co_matrix[j, i] = float(len(user_intersect))/float(len(user_union))
                else:
                    co_matrix[j, i] = 0
        
        return co_matrix

    def get_recommendation(self, user, cooccurence, all_songs, user_songs):
        user_sim_scores = cooccurence.sum(axis=0)/float(cooccurence.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        print(sort_index[:100])
        print(all_songs[sort_index[0][1]])
        columns = ['user_id', 'song_id', 'song_name', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
         
        song_name_list = self.get_all_song_name()

        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]], song_name_list[sort_index[i][1]], sort_index[i][0],rank]
                rank += 1

        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def recommend(self, user, n):
        
        user_songs = self.get_user_songs(user)    
        print("No. of unique songs for the user: %d" % len(user_songs))
        all_songs = self.get_all()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix = self.set_coocurrence(user_songs, all_songs)
        
        df_recommendations = self.get_recommendation(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations[:n]
    
    def get_similar_items(self, item_list):
        
        user_songs = item_list
        
        all_songs = self.get_all()
        
        cooccurence_matrix = self.set_coocurrence(user_songs, all_songs)
        
        user = ""
        df_recommendations = self.get_recommendation(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations

def main():
    raw_uid = 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'
    dir_data = './collaborative_filtering/cf_data'
    recommend_number = 10
    file_path = '{}/dataset_user_5.txt'.format(dir_data)
    file_song_path = '{}/dataset_song_5.txt'.format(dir_data)
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)
    
    db = pymysql.connect("localhost",
                         "root",
                         "",
                         "music_recommender",
                         charset='utf8')

    cursor = db.cursor()

    sql = """SELECT uid, song_id, rating
              FROM user_rating
               WHERE 1"""
    cursor.execute(sql)
    results = cursor.fetchall()
    with open(file_path, "w+", encoding="utf-8") as data_f:
        for result in results:
            uid, song_id, rating = result

            data_f.writelines("{}\t{}\t{}\n".format(uid, song_id, rating))
    
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    data_user = pd.read_csv(file_path,header=None,sep='\t',dtype={'song_id': object})
    data_user.columns = ["user_id", "song_id", "rating"]
    types_dict = {'user_id': str, 'song_id':str, 'rating': float}
    for col, col_type in types_dict.items():
        data_user[col] = data_user[col].astype(col_type)

    if not os.path.exists(file_path):
        raise IOError("Dataset file is not exists!")
    # file_path = ""

    sql_song = """SELECT song_id, song_name, artist_name,
                    album_id, album_name
                    FROM song_information
                    WHERE 1"""

    cursor.execute(sql_song)
    results_song = cursor.fetchall()
    # print(results_song)
    with open(file_song_path, "w+", encoding='utf-8') as data_f:
        for result in results_song:
            song_id, song_name, artist_name, album_id, album_name = result
            # print(result)
            data_f.writelines("{}\t{}\t{}\t{}\t{}\n".format(song_id, song_name, artist_name, album_id, album_name))

    # reader = Reader(line_format='song_id song_name artist_name album_name', sep='\t')
    data_song = pd.read_csv(file_song_path,header=None, sep='\t', dtype={'song_id': str})
    data_song.columns = ["song_id", "song_name", "artist_name", "album_id", "album_name"]

    if not os.path.exists(file_song_path):
        raise IOError("Dataset file is not exists!")
    # file_path = ""

    song_df = pd.merge(data_user, data_song.drop_duplicates(['song_id']), on="song_id", how="left")

    train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

    ib_model = ItemBasedRecommender()
    # print(song_df)
    ib_model.read_data(train_data, "user_id", "song_id")

    user_items = ib_model.get_user_songs(raw_uid)
    
    # print("------------------------------------------------------------------------------------")
    # print("Training data songs for the user userid: %s:" % raw_uid)
    # print("------------------------------------------------------------------------------------")

    # for user_item in user_items:
    #     print(user_item)

    print("----------------------------------------------------------------------")
    print("Recommendation process going on:")
    print("----------------------------------------------------------------------")

    #Recommend songs for the user using personalized model
    result = ib_model.recommend(raw_uid, recommend_number)
    print(result)

if __name__ == "__main__":
    main()