import json
import os
from datetime import datetime as dt


class DataSerializer():
    def __init__(self, file_name):
        data_path = os.getcwd() + '/data/'
        f = open(data_path + file_name, 'r')
        self.datas = json.load(f)
        self.texts = []

    def text_datas(self):
        texts = []
        for data in self.datas:
            if data['is_retweet'] == False or data['in_reply_to_user_id_str'] == None:
                parsed_time = dt.strptime(
                    data['created_at'], '%a %b %d %H:%M:%S %z %Y')
                texts.append(
                    {'text': data['text'], 'date': parsed_time})
        self.texts = texts
        return texts

    def limit_data_with_date(self, max_time):
        return list(filter(lambda text: text['date'] < max_time, self.texts))
