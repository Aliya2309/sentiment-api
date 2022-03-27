from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import requests
import os
import json

app = Flask(__name__)
api = Api(app)

with open('classifierpickle.pkl','rb') as file:
    mp = pickle.load(file)


bearer_token = "AAAAAAAAAAAAAAAAAAAAACIlZAEAAAAA%2Fg6vHg2fnRvrb9N6tgJA%2BnyrvlA%3DxOW8pS59BsTq4QThc3Cm7DOwVhJjnAMdGX4BmGk0CrbCL96T6m"

search_url = "https://api.twitter.com/2/tweets/search/recent"

#parser = reqparse.RequestParser()
#parser.add_argument('query')
def bearer_oauth(r):

        r.headers["Authorization"] = f"Bearer {bearer_token}"
        r.headers["User-Agent"] = "v2RecentSearchPython"
        return r

def connect_to_endpoint(url, params):
        response = requests.get(url, auth=bearer_oauth, params=params)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


class PredictSentiment(Resource):




    def get(self, query1):
        # use parser and find the user's query
       
        #args = parser.parse_args()
        #user_query = 'eggs'
        #print(args)
        #request.get_json(force=True)
        #user_query = args['query']
        
        query_params = {'query': query1,'tweet.fields': 'author_id'}
        
        json_response = connect_to_endpoint(search_url, query_params)
        
        xyz = list(json_response.values())[0]
        tweets = []
        ans = 0
        pos_score = 0
        neg_score = 0

        for i in xyz:
            tweets.append(i['text'])

        for t in tweets:
            ans = mp.predict([t])
            if ans == 0:
                neg_score = neg_score + 1
            elif ans == 4:
                pos_score = pos_score + 1

        score = pos_score / (pos_score + neg_score)
        

        # create JSON object
        output = {'prediction': score, 'args': tweets }

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/<string:query1>')


if __name__ == '__main__':
    app.run(debug=True)