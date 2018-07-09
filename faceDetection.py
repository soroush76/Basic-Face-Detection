
# coding: utf-8

# In[106]:


# !apt-get install -y -qq software-properties-common python-software-properties module-init-tools

# !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null

# !apt-get update -qq 2>&1 > /dev/null

# !apt-get -y install -qq google-drive-ocamlfuse fuse

# from google.colab import auth

# auth.authenticate_user()


# from oauth2client.client import GoogleCredentials

# creds = GoogleCredentials.get_application_default()

# import getpass

# !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL

# vcode = getpass.getpass()

# !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


# !mkdir -p drive

# !google-drive-ocamlfuse drive


# In[107]:


# !ls drive/ColabNotebooks/


# In[108]:


import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


# In[124]:


def initialize_parameters(n_input, n_hidden, solver): # initialize parameters with tiny random numbers
    num_of_layers = len(n_hidden)
    parameters = {}
    v = {}
    s = {}
    
    parameters['w1'] = np.random.randn(n_hidden[0], n_input) * np.sqrt(2/n_input)
    parameters['b1'] = np.zeros((n_hidden[0], 1))
    
    if solver=='momentum' or solver=='adam':
        v['dw1'] = np.zeros(parameters['w1'].shape)
        v['db1'] = np.zeros(parameters['b1'].shape)
    if solver=='rmsprop' or solver=='adam':
        s['dw1'] = np.zeros(parameters['w1'].shape)
        s['db1'] = np.zeros(parameters['b1'].shape)
    
    for i in range(1, num_of_layers):
        parameters['w'+str(i+1)] = np.random.randn(n_hidden[i], n_hidden[i-1]) * np.sqrt(2/n_hidden[i-1])
        parameters['b'+str(i+1)] = np.zeros((n_hidden[i], 1))
        
        if solver=='momentum' or solver=='adam':
            v['dw'+str(i+1)] = np.zeros(parameters['w'+str(i+1)].shape)
            v['db'+str(i+1)] = np.zeros(parameters['b'+str(i+1)].shape)
        if solver=='rmsprop' or solver=='adam':
            s['dw'+str(i+1)] = np.zeros(parameters['w'+str(i+1)].shape)
            s['db'+str(i+1)] = np.zeros(parameters['b'+str(i+1)].shape)
    
    parameters['w'+str(num_of_layers+1)] = np.random.randn(1, n_hidden[num_of_layers-1]) * np.sqrt(2/n_hidden[num_of_layers-1])
    parameters['b'+str(num_of_layers+1)] = np.zeros((1, 1))

    if solver=='momentum' or solver=='adam':
        v['dw'+str(num_of_layers+1)] = np.zeros(parameters['w'+str(num_of_layers+1)].shape)
        v['db'+str(num_of_layers+1)] = np.zeros(parameters['b'+str(num_of_layers+1)].shape)
    if solver=='rmsprop' or solver=='adam':
        s['dw'+str(num_of_layers+1)] = np.zeros(parameters['w'+str(num_of_layers+1)].shape)
        s['db'+str(num_of_layers+1)] = np.zeros(parameters['b'+str(num_of_layers+1)].shape)
    
    if solver=='momentum':
        return parameters, v
    elif solver=='rmsprop':
        return parameters, s
    elif solver=='adam':
        return parameters, v, s
    
    return parameters


# In[110]:


def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    der = np.zeros(z.shape)
    der[z >= 0] = 1
    return der


# In[111]:


def forward_prop(x, parameters, activation, keep_prob=1.0):
    cache = {}
    last_layer = len(parameters) // 2
    cache['a'+str(0)] = x
    
    for l in range(1, last_layer+1):
        cache['z'+str(l)] = np.dot(parameters['w'+str(l)], cache['a'+str(l-1)]) + parameters['b'+str(l)]
        if l == last_layer: 
            cache['a'+str(last_layer)] = 1/(1 + np.exp(-cache['z'+str(last_layer)]))
        elif activation == 'relu':
            cache['a'+str(l)] = relu(cache['z'+str(l)])
        elif activation == 'sigmoid': 
            cache['a'+str(l)] = 1/(1 + np.exp(-cache['z'+str(l)]))
        elif activation == 'tanh': 
            cache['a'+str(l)] = np.tanh(cache['z'+str(l)])
        
        if keep_prob != 1.0 and l != last_layer:
            dropout_temp = np.random.randn(cache['a'+str(l)].shape[0], cache['a'+str(l)].shape[1]) < keep_prob
            cache['dropout_vec'+str(l)] = dropout_temp 
            cache['a'+str(l)] = np.multiply(cache['a'+str(l)], dropout_temp) / keep_prob

    return cache


# In[112]:


def back_prop(x, y, cache, parameters, activation, _lambda, keep_prob):
    m = x.shape[1]
    gradients = {}
    last_layer = len(parameters) // 2

    gradients['dz'+str(last_layer)] = cache['a'+str(last_layer)] - y.T
    gradients['dw'+str(last_layer)] = 1/m * np.dot(gradients['dz'+str(last_layer)], cache['a'+str(last_layer-1)].T) + _lambda/m * parameters['w'+str(last_layer)]
    gradients['db'+str(last_layer)] = 1/m * np.sum(gradients['dz'+str(last_layer)], axis=1, keepdims=True)
        
    for i in reversed(range(1, last_layer)):
        dA = np.dot(parameters['w'+str(i+1)].T, gradients['dz'+str(i+1)])
        if keep_prob != 1.0 and i != last_layer:
            dA *= cache['dropout_vec'+str(i)]
            dA /= keep_prob
            
        if activation == 'sigmoid': 
            sigmoid = 1/(1 + np.exp(-cache['z'+str(i)]))
            gradients['dz'+str(i)] = dA * (sigmoid * (1 - sigmoid))
        elif activation == 'relu':
            gradients['dz'+str(i)] = dA * relu_derivative(cache['z'+str(i)])
        elif activation == 'tanh': 
            gradients['dz'+str(i)] = dA * (1 - np.power(np.tanh(cache['z'+str(i)]), 2))
            
        gradients['dw'+str(i)] = 1/m * np.dot(gradients['dz'+str(i)], cache['a'+str(i-1)].T) + _lambda/m * parameters['w'+str(i)]
        gradients['db'+str(i)] = 1/m * np.sum(gradients['dz'+str(i)], axis=1, keepdims=True)
        
    return gradients


# In[113]:


def cost_function(m, labels, y_hat, params, lambd): # regularized
    last_layer = len(params) // 2
    return -1/m * (np.sum(labels * np.log(y_hat)) + np.sum((1-labels) * np.log(1 - y_hat))) + lambd/(2*m) * np.sum([
            np.power(np.linalg.norm(params['w'+str(i)]), 2) for i in range(1, last_layer+1)])


# In[114]:


def load_weights(num):
    parameters = {}
    for i in range(1, num+1):
        parameters["w"+str(i)] = np.load("w"+str(i)+".npy")
        parameters["b"+str(i)] = np.load("b"+str(i)+".npy")
    return parameters


# In[115]:


params = load_weights(4)


# In[126]:


def multi_Layer_NN(samples, labels, n_hidden, num_iterations, solver='gd', batch_size=None, activation="tanh", learning_rate=0.01, momentum_param=0.9, rmsprop_param=0.999, epsilon=10e-8, _lambda=0, keep_prob_dropout=1.0, print_cost=False):
    m = samples.shape[1]
    if not batch_size: batch_size = m
    
    if solver=='gd':
        params = initialize_parameters(samples.shape[0], n_hidden, solver)
    elif solver=='momentum':
        params, v = initialize_parameters(samples.shape[0], n_hidden, solver)
    elif solver=='rmsprop':
        params, s = initialize_parameters(samples.shape[0], n_hidden, solver)
    elif solver=='adam':
        params, v, s = initialize_parameters(samples.shape[0], n_hidden, solver)
            
    cost_history = []
    last_layer = len(params) // 2
    
    for i in range(1, num_iterations+1):
        
        for t in range(math.ceil(m/batch_size)):
            
            cache = forward_prop(samples[:, int(t*batch_size):int((t+1)*batch_size)], params, activation, keep_prob_dropout)
            
            gradients = back_prop(samples[:, int(t*batch_size):int((t+1)*batch_size)],
                                  labels[int(t*batch_size):int((t+1)*batch_size)], 
                                  cache, 
                                  params, 
                                  activation,
                                  _lambda, 
                                  keep_prob_dropout)
            
            if solver=='gd':
                for j in range(1, last_layer+1):
                    params['w'+str(j)] -= learning_rate * gradients['dw'+str(j)]
                    params['b'+str(j)] -= learning_rate * gradients['db'+str(j)]
            elif solver=='momentum':
                for j in range(1, last_layer+1):
                    v['dw'+str(j)] = momentum_param*v['dw'+str(j)] + (1-momentum_param) * gradients['dw'+str(j)]
                    v['db'+str(j)] = momentum_param*v['db'+str(j)] + (1-momentum_param) * gradients['db'+str(j)]
                    params['w'+str(j)] -= learning_rate * v['dw'+str(j)]
                    params['b'+str(j)] -= learning_rate * v['db'+str(j)]
            elif solver=='rmsprop':
                for j in range(1, last_layer+1):
                    s['dw'+str(j)] = rmsprop_param*s['dw'+str(j)] + (1-rmsprop_param) * gradients['dw'+str(j)]**2
                    s['db'+str(j)] = rmsprop_param*s['db'+str(j)] + (1-rmsprop_param) * gradients['db'+str(j)]**2
                    params['w'+str(j)] -= learning_rate * gradients['dw'+str(j)]/(np.sqrt(s['dw'+str(j)] + epsilon))
                    params['b'+str(j)] -= learning_rate * gradients['db'+str(j)]/(np.sqrt(s['db'+str(j)] + epsilon))
            
            elif solver=='adam':
                for j in range(1, last_layer+1):
                    v['dw'+str(j)] = momentum_param*v['dw'+str(j)] + (1-momentum_param) * gradients['dw'+str(j)]
                    v['db'+str(j)] = momentum_param*v['db'+str(j)] + (1-momentum_param) * gradients['db'+str(j)]
                    s['dw'+str(j)] = rmsprop_param*s['dw'+str(j)] + (1-rmsprop_param) * gradients['dw'+str(j)]**2
                    s['db'+str(j)] = rmsprop_param*s['db'+str(j)] + (1-rmsprop_param) * gradients['db'+str(j)]**2
                    v_corrected_dw = v['dw'+str(j)]/(1-momentum_param**i)
                    v_corrected_db = v['db'+str(j)]/(1-momentum_param**i)
                    s_corrected_dw = s['dw'+str(j)]/(1-rmsprop_param**i)
                    s_corrected_db = s['db'+str(j)]/(1-rmsprop_param**i)
                    params['w'+str(j)] -= learning_rate * v_corrected_dw/(np.sqrt(s_corrected_dw) + epsilon)
                    params['b'+str(j)] -= learning_rate * v_corrected_db/(np.sqrt(s_corrected_db) + epsilon)
            
            if print_cost: 
                cost = cost_function(batch_size, labels[int(t*batch_size):int((t+1)*batch_size)], cache['a'+str(last_layer)].T, params, _lambda)
                print('cost after epoch {}: {}'.format(int(i), cost))
                cost_history.append(cost)
              
    return {'parameters':params, 
            'cache': cache,
            'cost_history':cost_history
           }


# In[117]:


def predict(x, parameters, activation):
    last_layer = len(parameters) // 2
    cache = forward_prop(x.T, parameters, activation)
    y_hat = cache['a'+str(last_layer)]
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0

    return y_hat.T


# ## load dataset section

# In[118]:


import scipy.io
samples = scipy.io.loadmat('olivettifaces.mat')
samples = samples['faces'].T
labels = np.ones((samples.shape[0], 1))
artificialZeroData = np.random.randint(0, 256, (samples.shape[0], samples.shape[1]))
artificialZeroLabel = np.zeros((artificialZeroData.shape[0], 1))
samples = np.vstack((samples, artificialZeroData))
labels = np.vstack((labels, artificialZeroLabel))

_mean = np.mean(samples, axis=1).reshape(-1, 1)
variance = (np.std(samples, axis=1)**2).reshape(-1, 1)
samples = (samples - _mean)/variance

samples_sparse = coo_matrix(samples)
samples, samples_sparse, labels = shuffle(samples, samples_sparse, labels)

train_data, test_data, train_label, test_label = train_test_split(samples, labels, test_size=0.30, random_state=4)


# In[119]:


print(samples.shape, labels.shape)
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)


# ## train section

# In[131]:


import time

tic = time.time()
activation_function = 'tanh'
model = multi_Layer_NN(train_data.T,
                         train_label,
                         activation=activation_function,
                         n_hidden=[100, 100, 100],
                         num_iterations=50,
                         solver='adam',
                         momentum_param=0.9,
                         rmsprop_param=0.999,
                         epsilon=10e-2,
                         batch_size=64,
                         learning_rate=0.1,
                         _lambda = 0,
                         keep_prob_dropout=1,
                         print_cost=True)

print("Train phase time:", time.time()-tic)


# In[132]:


pred_labels = predict(train_data, parameters=model['parameters'], activation=activation_function)
print('accuracy on train set:', (np.sum(pred_labels == train_label)/pred_labels.size) * 100, '%')
pred_labels = predict(test_data, parameters=model['parameters'], activation=activation_function)
print('accuracy on test set:', (np.sum(pred_labels == test_label)/pred_labels.size) * 100, '%')


# ### plot cost value

# In[133]:


plt.plot(range(len(model['cost_history'])), model['cost_history'])
plt.show()


# ### test an example of test set

# In[141]:


photo_number = np.random.randint(0, test_data.shape[0])
imgplot = plt.imshow(test_data[photo_number].reshape(64, 64))
plt.show()
pred_labels = predict(test_data[photo_number].reshape(1, -1), parameters=params, activation='tanh')
print(pred_labels)
if int(pred_labels[0][0]) == 1:
    print('Human')
else:
    print('Not-Human')


# ## Save the model in files

# In[ ]:


def save_weights(parameters):
    for i in range(1, (len(parameters)//2)+1):
        np.save("drive/ColabNotebooks/w"+str(i), parameters['w'+str(i)])
        np.save("drive/ColabNotebooks/b"+str(i), parameters['b'+str(i)])


# In[ ]:


save_weights(model['parameters'])


# ### predict an example from you
# It must be a 64*64 photo.

# In[58]:


# myimg = mpimg.imread('myimg.jpg')
# imgplot = plt.imshow(myimg)
# print(myimg.shape)

def predict_new_photo(photo_name):
    # import photo and convert it to greyscale 64*64
    x=Image.open(photo_name, 'r')
    x=x.convert('L') #makes it greyscale
    y=np.asarray(x.getdata(), dtype=np.float64).reshape((x.size[1],x.size[0]))
    y=np.asarray(y, dtype=np.uint8) #if values still in range 0-255! 
    w=Image.fromarray(y,mode='L')
    w.save('out.jpg')

    # rotate imgae 90 degree
    raw_img = Image.open("out.jpg")
    img = raw_img.rotate(90)
    # img.show()
    img.save("out.jpg")

    myimg = mpimg.imread('out.jpg')
    imgplot = plt.imshow(myimg)
    
    pred_labels = predict(myimg.reshape(1, -1), parameters=params, activation='tanh')
    if int(pred_labels[0][0]) == 1:
        plt.text(15,-3,'Human')
    else:
        plt.text(15,-3,'Not-Human')


# In[61]:


photos = ['myimg.jpg', 'myimg2.jpg', 'myimg3.jpg', 'myimg4.jpg', 'myimg5.jpg', 'myimg6.jpg', 'myimg7.jpg', 'myimg8.jpg', 'myimg9.jpg', 'myimg10.jpg', 'myimg11.jpg', 'myimg12.jpg', 'myimg13.jpg']
for i in range(len(photos)):
    plt.subplot(4,4, i+1)
    predict_new_photo(photos[i])


# ### run a package with the same parameters to check it

# In[ ]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(200, 500, 200), activation='tanh', solver='sgd', alpha=0, learning_rate_init=0.1, max_iter=10000)
clf.fit(train_data, train_label)

print('accuracy on train set:', clf.score(train_data, train_label)*100)
print('accuracy on test set:', clf.score(test_data, test_label)*100)
print('loss:', clf.loss_)


# In[ ]:


myimg = mpimg.imread('out.jpg')
print(clf.predict(myimg.reshape(1, -1)))

