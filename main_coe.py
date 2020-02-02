# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:48:57 2020

@author: shris
"""
#imoprt numpy and lightfm
#pip install lightfm
#conda install -c conda-forge lightfm (run as administrator)

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data from model with minimun rating of 4
data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss = 'warp')

#train mode
model.fit(data['train'], epochs=30, num_threads=2)

#recommender fucntion
def sample_recommendation(model, data, user_ids):
    #number of users and movies in training data
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
    	#movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #sort them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
            
#show 3 positives and 3 recommended movies for the specified users             
sample_recommendation(model, data, [300, 215, 451,8,98,134,789])




