Detection  SQL Injection  with Deep Learning 

环境

> tensorflow

> keras
说明


1. data目录下是使用的数据，包括一个sql注入样本和normal正常样本，已经额外测试样本。
2. file目录保存训练好的词向量、预处理的数据、训练好的模型等。
3. log目录保存训练日志，可用tensorborad可视化。
运行
1. 运行word2vec.py训练嵌入式词向量
2. 运行processing.py预处理数据，生成训练数据和测试数据。
3. MLP.py、LSTM.py、Conv.py分别使用多层感知机、长短时记忆、卷积神经网络训练模型，在测试集上准确率和召回率，测试SVM上的准确率和召回率，并且用额外测试样本进一步测试。
