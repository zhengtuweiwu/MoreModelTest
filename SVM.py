import time
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from utils import GeneSeg
import csv,random,pickle


batch_size=50
maxlen=15#归一化长度
vec_dir="file\\word2vec.pickle"
epochs_num=1
model_dir="file\\SVM"
def data_process(sql,normal,scale):
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
    datas_index=pad_sequences(datas_index,value=-1,maxlen=maxlen)
    rand=random.sample(range(len(datas_index)),len(datas_index))
    datas=[datas_index[index] for index in rand]
    labels=[labels[index] for index in rand]
    datas_embed=[]
    dims=len(embeddings[0])

    for data in datas:
        data_embed = []
        for d in data:
            if d != -1:
                data_embed.extend(embeddings[d])
            else:
                data_embed.extend([0.0] * dims)
        datas_embed.append(data_embed)

    train_datas,test_datas,train_labels,test_labels=train_test_split(datas_embed,labels,test_size=scale)
    return train_datas,test_datas,train_labels,test_labels
if __name__=="__main__":
    train_datas, test_datas, train_labels, test_labels=data_process("data\\sql-original-noshort-norepeat-sqlfilter.txt","data\\normal-original-noshort-norepeat-sqlfilter.txt",0.1)
    more_test_datas, kong1, more_test_labels, kong2 = data_process("data\\test-sql.txt", "data\\test-sql-like.txt",0.0)
    print("Start Train! ")
    start = time.time()
    model=LinearSVC()
  #  model = SVC(C=1.0, kernel="linear")
    model.fit(train_datas,train_labels)
    end = time.time()
    print("Over train job in %f s" % (end - start))
    print("Start Test!")
    start=time.time()
    pre=model.predict(test_datas)
    end=time.time()
    print("Over test job in %s s"%(end-start))
    precision = precision_score(test_labels, pre)
    recall = recall_score(test_labels, pre)
    print("Precision score is :", precision)
    print("Recall score is :", recall)
    pre = model.predict(more_test_datas)
    end = time.time()
    print("Over more test job in %s s" % (end - start))
    precision = precision_score(more_test_labels, pre)
    recall = recall_score(more_test_labels, pre)
    print("More precision score is :", precision)
    print("More recall score is :", recall)
    with open(model_dir,"wb") as f:
        pickle.dump(model,f,protocol=2)
    print("wirte to ",model_dir)