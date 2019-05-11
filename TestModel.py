import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split
import keras
from utils import GeneSeg
import csv,random,pickle
import numpy as np
from keras.utils import to_categorical
from utils import init_session
init_session()
batch_size=50
maxlen=189#归一化长度
vec_dir="file\\word2vec.pickle"
epochs_num=1
hidden_layer=2
model_dir="file\\"
#def data_process(sql,normal,scale):
def data_process(sql, normal):
    with open(vec_dir,"rb") as f :
        word2vec=pickle.load(f)
        dictionary=word2vec["dictionary"]
        embeddings=word2vec["embeddings"]
        reverse_dictionary = word2vec["reverse_dictionary"]
    sql_data=[]
    normal_data=[]
    with open(sql,"r",encoding="utf-8") as f:
        reader = csv.DictReader((line.replace('\0','') for line in f), fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            if len(word)<=maxlen:
                sql_data.append(word)
    with open(normal,"r",encoding="utf-8") as f:
        reader = csv.DictReader((line.replace('\0','') for line in f), fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            if len(word) <= maxlen:
                normal_data.append(word)
    sql_num=len(sql_data)
    normal_num=len(normal_data)
    sql_labels=[1]*sql_num
    normal_labels=[0]*normal_num
    datas=sql_data+normal_data
    labels=sql_labels+normal_labels
    def to_index(data):
        d_index=[]
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index
    datas_index=[to_index(data) for data in datas]
    datas=pad_sequences(datas_index,value=-1,maxlen=maxlen)
    #rand=random.sample(range(len(datas_index)),len(datas_index))  ###随机打乱顺序
    #datas=[datas[index] for index in rand]
    #labels=[labels[index] for index in rand]
    datas_embed=[]
    datas_labels=[]
    dims=len(embeddings[0])
    #print(labels)
    datas_labels = to_categorical(labels)
    #print(datas_labels)
    for data in datas:
        data_embed = []
        for d in data:
            if d != -1:
                data_embed.append(embeddings[d])
            else:
                data_embed.append([0.0] * dims)
        datas_embed.append(data_embed)
    #train_datas,test_datas,train_labels,test_labels=train_test_split(datas_embed,datas_labels,test_size=scale)
    #print(train_labels)
    return np.array(datas_embed),datas_labels
if __name__=="__main__":
    #more_test_datas, kong1, more_test_labels, kong2 = data_process("data\\test-sql.txt", "data\\test-sql-like.txt",0.0)
    more_test_datas, more_test_labels = data_process("data\\test-sql.txt", "data\\test-sql-like.txt")
    #with open('file\\GRU-'+str(hidden_layer)+'.json', 'r') as file:
    with open('file\\LSTM-384-3-2.json', 'r') as file:
        json_string = file.read()
    new_model = keras.models.model_from_json(json_string)
    new_model.load_weights('file\\LSTM-384-3-2.h5', by_name=True)
    #new_model.load_weights('file\\GRU-' + str(hidden_layer) + '.h5', by_name=True)
    labels_pre = []
    #labels_true = []
    print("Start Test!")
    start=time.time()
    labels_pre.extend(new_model.predict_on_batch(more_test_datas))
    #print(labels_pre)
    labels_pre = np.array(labels_pre).round()
    #print(labels_pre)
    #labels_true.extend(more_test_labels)
    #print(labels_true)
    def to_y(labels):
        y = []
        for i in range(len(labels)):
            if labels[i][0] == 1:
                y.append(0)
            else:
                y.append(1)
        return y

    y_true = to_y(more_test_labels)
    y_pre = to_y(labels_pre)
    print(y_pre)
    end = time.time()
    print("Over more test job in %s s" % (end - start))
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    print("More precision score is :", precision)
    print("More recall score is :", recall)
