import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from sklearn.cluster import KMeans
import metrics1
import tensorflow as tf
from tensorflow import keras
import pandas as pd

balance = pd.read_table("balance.data", sep=",", header=None)
data= balance.iloc[:, 1:6].values
label=balance.iloc[:,0].values#3
label[np.reshape(np.argwhere(label=='L'),-1)]=[0]*np.sum(label=='L')
label[np.reshape(np.argwhere(label=='B'),-1)]=[1]*np.sum(label=='B')
label[np.reshape(np.argwhere(label=='R'),-1)]=[2]*np.sum(label=='R')
dim=data.shape[1]
k=max(label)-min(label)+1
min_max_scaler = MinMaxScaler()
data=min_max_scaler.fit_transform(data)

centers=tf.Variable(tf.random_normal([k,dim]))
y=tf.Variable(data,dtype=tf.float32)
q = 1.0 / (1 + (tf.reduce_sum(tf.square(tf.expand_dims(y, axis=1) - centers), axis=2) / 1.))
q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
pp=tf.placeholder(dtype=tf.float32,shape=(None,k))
loss=keras.losses.kullback_leibler_divergence(pp,q)
train=tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full').fit(data)
    centers_g = gmm.means_
    sess.run(tf.assign(centers, centers_g))

    y_pro = sess.run(q)
    y_pred = np.argmax(y_pro, 1)

    num_clu = [np.sum(y_pred == j) for j in range(k)]
    num_clu = np.array(num_clu)
    num_clu = num_clu / np.sum(num_clu)

    pro_clu = np.sum(y_pred, 0)
    pro_clu = pro_clu / np.sum(pro_clu)

    reg_clu = num_clu * pro_clu
    reg_clu = reg_clu / np.sum(reg_clu)
    for i in range(400):
        y_pred = sess.run(q)
        weight = (y_pred ** 2 / tf.reduce_sum(y_pred, 0))*reg_clu
        p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, 1))
        p = sess.run(p)
        y_pred = sess.run(tf.argmax(y_pred, 1))
        print("epoch:"+str(i))
        print("ACC:"+str(metrics1.acc(label, y_pred)))
        print("ARI:"+str(metrics.adjusted_rand_score(label, y_pred)))
        print("NMI:"+str(metrics.normalized_mutual_info_score(label, y_pred)))
        sess.run(train, feed_dict={pp: p})