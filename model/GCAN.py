
# coding: utf-8

# In[ ]:


##graph convolution network
import scipy.sparse as sp
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def preprocess_adj_numpy(adj, symmetric=True):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj


def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor

def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor

import keras.backend as K
import tensorflow as tf


def graph_conv_op(x, num_filters, graph_conv_filters, kernel):

    if len(x.get_shape()) == 2:
        conv_op = K.dot(graph_conv_filters, x)
        conv_op = tf.split(conv_op, num_filters, axis=0)
        conv_op = K.concatenate(conv_op, axis=1)
    elif len(x.get_shape()) == 3:
        conv_op = K.batch_dot(graph_conv_filters, x)
        conv_op = tf.split(conv_op, num_filters, axis=1)
        conv_op = K.concatenate(conv_op, axis=2)
    else:
        raise ValueError('x must be either 2 or 3 dimension tensor'
                         'Got input shape: ' + str(x.get_shape()))

    conv_out = K.dot(conv_op, kernel)
    return conv_out

from keras import activations, initializers, constraints
from keras import regularizers
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf



class MultiGraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 num_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiGraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.num_filters = num_filters

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        if self.num_filters != int(input_shape[1][-2]/input_shape[1][-1]):
            raise ValueError('num_filters does not match with graph_conv_filters dimensions.')

        self.input_dim = input_shape[0][-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):

        output = graph_conv_op(inputs[0], self.num_filters, inputs[1], self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], self.output_dim)
        return output_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_filters': self.num_filters,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MultiGraphCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    


##Dual Co-Attention: source tweet and graph-aware interaction embeddings
from keras import backend as k
from keras.engine.topology import Layer

class coattention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(coattention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        
        self.kernelW = self.add_weight(name='Wall', 
                                      shape=(source tweet output dim, GCN output dim),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelWs = self.add_weight(name='Ws', 
                                      shape=(rewteet user size,rewteet user size),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelWc = self.add_weight(name='Wc', 
                                      shape=(source tweet length,source tweet length),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelas = self.add_weight(name='Was', 
                                      shape=(GCN output dim,1),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelac = self.add_weight(name='Wac', 
                                      shape=(source tweet output dim,1),
                                      initializer='uniform',
                                      trainable=True)
        super(coattention, self).build(input_shape)  


    def call(self, x):
        C=x[0]
       
        print("C.shape",C.shape)
        RNN=Permute((2,1))(x[1])
        
        f=k.dot(C,self.kernelW)
        print("f.shape",f.shape)
        F=k.tanh(k.batch_dot(f,RNN))
        print("F.shape",F.shape)
        
        s=k.dot(RNN,self.kernelWs)
        print("s.shape",s.shape)
        c=k.dot(Permute((2,1))(C),self.kernelWc)
        print("c.shape",c.shape)
       
        Hs=k.tanh(s+k.batch_dot(c,F))
        print("Hs.shape",Hs.shape)
        Hc=k.tanh(c+k.batch_dot(s,Permute((2,1))(F)))
        print("Hc.shape",Hc.shape)
        
        
        As=k.softmax(k.dot(Permute((2,1))(Hs),self.kernelas))
        print("As.shape",As.shape)
        Ac=k.softmax(k.dot(Permute((2,1))(Hc),self.kernelac))
        print("Ac.shape",Ac.shape)
        
        As=Permute((2,1))(As)
        print("As.shape",As.shape)
        Ac=Permute((2,1))(Ac)
        print("Ac.shape",Ac.shape)
        
        sfinal=k.batch_dot(As,Permute((2,1))(RNN))
        
        print("sfinal.shape",sfinal.shape)
        cfinal=k.batch_dot(Ac,C)
       
        print("cfinal.shape",cfinal.shape)
        
        return keras.layers.concatenate([sfinal,cfinal])

    def compute_output_shape(self, input_shape):
        return (None, 64)
    
##Dual Co-Attention: source tweet and user propagation embeddings
from keras import backend as k
from keras.engine.topology import Layer

class cocnnattention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(cocnnattention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        
        self.kernelW = self.add_weight(name='Wall', 
                                      shape=(source tweet output dim, cnn output dim),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelWs = self.add_weight(name='Ws', 
                                      shape=(cnn output length, cnn output length),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelWc = self.add_weight(name='Wc', 
                                      shape=(source tweet length, source tweet length),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelas = self.add_weight(name='Was', 
                                      shape=(cnn output dim,1),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelac = self.add_weight(name='Wac', 
                                      shape=(source tweet output dim,1),
                                      initializer='uniform',
                                      trainable=True)
        super(cocnnattention, self).build(input_shape)  


    def call(self, x):
        C=x[0]
       
        print("C.shape",C.shape)
        RNN=Permute((2,1))(x[1])
        
        f=k.dot(C,self.kernelW)
        print("f.shape",f.shape)
        F=k.tanh(k.batch_dot(f,RNN))
        print("F.shape",F.shape)
        
        s=k.dot(RNN,self.kernelWs)
        print("s.shape",s.shape)
        c=k.dot(Permute((2,1))(C),self.kernelWc)
        print("c.shape",c.shape)
       
        Hs=k.tanh(s+k.batch_dot(c,F))
        print("Hs.shape",Hs.shape)
        Hc=k.tanh(c+k.batch_dot(s,Permute((2,1))(F)))
        print("Hc.shape",Hc.shape)
        
        
        As=k.softmax(k.dot(Permute((2,1))(Hs),self.kernelas))
        print("As.shape",As.shape)
        Ac=k.softmax(k.dot(Permute((2,1))(Hc),self.kernelac))
        print("Ac.shape",Ac.shape)
        
        As=Permute((2,1))(As)
        Ac=Permute((2,1))(Ac)
        
        sfinal=k.batch_dot(As,Permute((2,1))(RNN))
        
        print("sfinal.shape",sfinal.shape)
        cfinal=k.batch_dot(Ac,C)
       
        print("cfinal.shape",cfinal.shape)
        
        return keras.layers.concatenate([sfinal,cfinal])

    def compute_output_shape(self, input_shape):
        return (None, 64)
    
    

