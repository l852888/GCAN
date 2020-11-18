
# coding: utf-8

# In[ ]:




#==================================================================================================#
#evaluation
def f1_score(y_true, y_pred):
    #calculate tp、tn、fp、fn
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
	
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def precision(y_true, y_pred):
    #calculate tp、tn、fp、fn
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
	
   
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    #calculate f1
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return p

def recall(y_true, y_pred):
    #calculate tp、tn、fp、fn
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
	
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    #calculate f1
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return r
#=======================================================================================#
##==start to build and train GCAN model==##

#data preparation
matrix=cos   #cos is calculated from data_preparation.py, use the user profile to calculate their cosine similarity for building the graph
graph_conv_filters=preprocess_adj_tensor(matrix)
y=pd.read_csv(r".csv")
y=y[0:dataset size]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data_all,y,test_size=0.3,random_state= )
WX_train,WX_test,y_train,y_test=train_test_split(padded_docs,y,test_size=0.3,random_state= )
MX_train,MX_test,y_train,y_test=train_test_split(graph_conv_filters,y,test_size=0.3,random_state=)
    
cnnX_train=np.reshape(X_train,(training size,retweet user size,number of feature,1))
cnnX_test=np.reshape(X_test,(testing size,retweet user size,number of feature,1))

from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train,2)
y_train= y_train.astype('int')
y_test= to_categorical(y_test,2)
y_test= y_test.astype('int')
    
num_filters = 1


#source tweet encoding
winput=Input(shape=(source_tweet_length,)) #source_tweet_length:in the paper is 40
wembed=Embedding(vocab_size,source_tweet_output_dim,input_length=source_tweet_length)(winput)
wembed=Reshape((source_tweet_length,source_tweet_output_dim))(wembed)  #source_tweet_output_dim: define by yourself
wembed=GRU(source_tweet_output_dim,return_sequences=True)(wembed)

#user propagation representation
rmain_input =Input(shape=(retweet_user_size,number of feature)) #number of feature: in the paper is 10, retweet_user_size: in the paper is 40
rnnencoder=GRU(output_dim,return_sequences=True)(rmain_input)
rnnoutput1= AveragePooling1D(retweet_user_size)(rnnencoder)
rnnoutput=Flatten()(rnnoutput1)

#Graph-aware Propagation Representation
graph_conv_filters_input = Input(shape=(retweet_user_size, retweet_user_size))
gmain_input= MultiGraphCNN(GCN_output_dim, num_filters)([rmain_input, graph_conv_filters_input])
gmain_input= MultiGraphCNN(GCN_output_dim, num_filters)([gmain_input, graph_conv_filters_input])

#dual co attention
gco=coattention(co_attention output dim)([wembed,gmain_input])
gco=Flatten()(gco)
    
cmain_input=Input(shape=(retweet_user_size,number of feature,1))
cnnco=Conv2D(cnn_output_dim,filter_size,number of feature,activation="sigmoid")(cmain_input)
maxpooling=Reshape((cnn_output_length,cnn_output_dim))(cnnco)
co=cocnnattention(co_attention output dim)([wembed,maxpooling])
co=Flatten()(co)

merged_vector=keras.layers.concatenate([co,gco,rnnoutput])
x=Dense(output_dim,activation="relu")(merged_vector)
x=Dense(output_dim,activation="relu")(x)
x=Dense(output_dim,activation="relu")(x)
prediction=Dense(2,activation="softmax")(x) #Here is output:class{0,1}
    
model=Model([winput,rmain_input,cmain_input, graph_conv_filters_input],prediction)
model.summary()
  
from keras import optimizers
Adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=Adam,loss="categorical_crossentropy",metrics=['accuracy', f1_score,precision,recall])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
history=model.fit([np.array(WX_train),np.array(X_train),np.array(cnnX_train),np.array(MX_train)],np.array(y_train),epochs=50,validation_split=0.1, callbacks=[early_stopping])
scores=model.evaluate([np.array(WX_test),np.array(X_test),np.array(cnnX_test),np.array(MX_test)],np.array(y_test), verbose=0)

