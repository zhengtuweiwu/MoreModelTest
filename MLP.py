import time
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from processing import build_dataset
import numpy as np
import keras
from utils import init_session
from sklearn.metrics import precision_score,recall_score
init_session()
batch_size=500
epochs_num=2
def train(train_generator,train_size,input_num,dims_num):
    print("Start Train! ")
    start=time.time()
    layer1=Dense(400,activation="relu",input_shape=(input_num,dims_num))
    layer2=Dense(200,activation="relu")
    flatten=Flatten()
    layer3=Dense(2,activation="softmax",name="Output")
    optimizer=Adam()
    model=Sequential()
    model.add(layer1)
    model.add(Dropout(0.5))
    model.add(layer2)
    model.add(Dropout(0.5))
    model.add(flatten)
    model.add(layer3)
    model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit_generator(train_generator,steps_per_epoch=train_size//batch_size,epochs=epochs_num)
    model.save_weights('file\\MLP.h5')
    model_json = model.to_json()
    with open('file\\MLP.json', 'w') as file:
        file.write(model_json)
    end = time.time()
    print("Over train job in %f s"%(end-start))
def test(test_generator,test_size,input_num,dims_num,batch_size):
    with open('file\\MLP.json', 'r') as file:
        json_string = file.read()
    new_model = keras.models.model_from_json(json_string)
    new_model.load_weights('file\\MLP.h5', by_name=True)
    labels_pre=[]
    labels_true=[]
    batch_num=test_size//batch_size+1
    steps=0
    for batch,labels in test_generator:
        if len(labels)==batch_size:
            labels_pre.extend(new_model.predict_on_batch(batch))
        else:
            batch=np.concatenate((batch,np.zeros((batch_size-len(labels),input_num,dims_num))))
            labels_pre.extend(new_model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps+=1
        print("%d/%d batch"%(steps,batch_num))
    labels_pre=np.array(labels_pre).round()
    def to_y(labels):
        y=[]
        for i in range(len(labels)):
            if labels[i][0]==1:
                y.append(0)
            else:
                y.append(1)
        return y
    y_true=to_y(labels_true)
    y_pre=to_y(labels_pre)
    precision=precision_score(y_true,y_pre)
    recall=recall_score(y_true,y_pre)
    print("Precision score is :",precision)
    print("Recall score is :",recall)
if __name__=="__main__":
    train_generator, test_generator, train_size, test_size, input_num, dims_num=build_dataset(batch_size)
    train(train_generator,train_size,input_num,dims_num)
    test(test_generator,test_size,input_num,dims_num,batch_size)

