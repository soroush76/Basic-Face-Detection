
# coding: utf-8

# In[ ]:


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


# In[ ]:


# !ls drive/ColabNotebooks/


# In[ ]:


import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


# In[ ]:


def load_data(): # load dataset from breast cancer file
    samples = []
    labels = []
    with open('breastcancer_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == 'M': labels.append(1)
            else: labels.append(0)

            samples.append([float(i) for i in row[2:]])
    samples, labels = np.array(samples), np.array(labels).reshape(-1, 1) # prevent returning a ranked-1-array
    samples = (samples - np.mean(samples, axis=1).reshape(-1, 1))/np.std(samples, axis=1).reshape(-1, 1)

    return samples, labels


# In[ ]:


def initialize_parameters(n_input, n_hidden): # initialize parameters with tiny random numbers
    num_of_layers = len(n_hidden)
    parameters = {}
    
    parameters['w1'] = np.random.randn(n_hidden[0], n_input)
    parameters['b1'] = np.random.randn(n_hidden[0], 1)
    
    for i in range(1, num_of_layers):
        parameters['w'+str(i+1)] = np.random.randn(n_hidden[i], n_hidden[i-1])
        parameters['b'+str(i+1)] = np.random.randn(n_hidden[i], 1)
    
    parameters['w'+str(num_of_layers+1)] = np.random.randn(1, n_hidden[num_of_layers-1])
    parameters['b'+str(num_of_layers+1)] = np.random.randn(1, 1)

    return parameters


# In[ ]:


def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    der = np.zeros(z.shape)
    der[z >= 0] = 1
    return der


# In[ ]:


def forward_prop(x, parameters, activation, keep_prop=1.0):
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
        
        if keep_prop != 1.0 and l != last_layer:
            dropout_temp = np.random.randn(cache['a'+str(l)].shape[0], cache['a'+str(l)].shape[1]) < keep_prop
            cache['a'+str(l)] = np.multiply(cache['a'+str(l)], dropout_temp) / keep_prop

    return cache


# In[ ]:


def back_prop(x, y, cache, parameters, activation, _lambda):
    m = x.shape[1]
    gradients = {}
    last_layer = len(parameters) // 2
    
    gradients['dz'+str(last_layer)] = cache['a'+str(last_layer)] - y.T
    gradients['dw'+str(last_layer)] = 1/m * np.dot(gradients['dz'+str(last_layer)], cache['a'+str(last_layer-1)].T) + _lambda/m * parameters['w'+str(last_layer)]
    gradients['db'+str(last_layer)] = 1/m * np.sum(gradients['dz'+str(last_layer)], axis=1, keepdims=True)
        
    for i in reversed(range(1, last_layer)):
        if activation == 'sigmoid': 
            sigmoid = 1/(1 + np.exp(-cache['z'+str(i)]))
            gradients['dz'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz'+str(i+1)]) * (sigmoid * (1 - sigmoid))
        elif activation == 'relu':
            gradients['dz'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz'+str(i+1)]) * relu_derivative(cache['z'+str(i)])
        elif activation == 'tanh': 
            gradients['dz'+str(i)] = np.dot(parameters['w'+str(i+1)].T, gradients['dz'+str(i+1)]) * (1 - np.power(np.tanh(cache['z'+str(i)]), 2))
     
        gradients['dw'+str(i)] = 1/m * np.dot(gradients['dz'+str(i)], cache['a'+str(i-1)].T) + _lambda/m * parameters['w'+str(i)]
        gradients['db'+str(i)] = 1/m * np.sum(gradients['dz'+str(i)], axis=1, keepdims=True)
    
    return gradients


# In[ ]:


def cost_function(m, labels, y_hat, params, lambd): # regularized
    last_layer = len(params) // 2
    return -1/m * (np.sum(labels * np.log(y_hat)) + np.sum((1-labels) * np.log(1 - y_hat))) + lambd/(2*m) * np.sum([
            np.power(np.linalg.norm(params['w'+str(i)]), 2) for i in range(1, last_layer+1)])


# In[ ]:


def multi_Layered_NN(samples, labels, n_hidden, num_iterations, activation="tanh", learning_rate=0.01, _lambda=0, keep_prob_dropout=1.0, print_cost=False):
    m = samples.shape[1]
    params = initialize_parameters(samples.shape[0], n_hidden)
    cost_history = []
    last_layer = len(params) // 2
    
    for i in range(1, num_iterations+1):
        cache = forward_prop(samples, params, activation, keep_prob_dropout)
        
        gradients = back_prop(samples, labels, cache, params, activation, _lambda)
        for j in range(1, last_layer+1):
            params['w'+str(j)] -= learning_rate * gradients['dw'+str(j)]
            params['b'+str(j)] -= learning_rate * gradients['db'+str(j)]

        if print_cost and i%1000 == 0: 
            cost = cost_function(m, labels, cache['a'+str(last_layer)].T, params, _lambda)
            print('cost after epoch {}: {}'.format(int(i/1000), cost))
            cost_history.append(cost)

    return {'parameters':params, 
            'cache': cache,
            'cost_history':cost_history
           }


# In[ ]:


def predict(x, parameters, activation):
    last_layer = len(parameters) // 2
    cache = forward_prop(x.T, parameters, activation)
    y_hat = cache['a'+str(last_layer)]
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0

    return y_hat.T


# ## load dataset section

# In[ ]:


import scipy.io
samples = scipy.io.loadmat('drive/ColabNotebooks/olivettifaces.mat')
samples = samples['faces'].T
labels = np.ones((samples.shape[0], 1))
artificialZeroData = np.random.randint(0, 256, (samples.shape[0], samples.shape[1]))
artificialZeroLabel = np.zeros((artificialZeroData.shape[0], 1))
samples = np.vstack((samples, artificialZeroData))
labels = np.vstack((labels, artificialZeroLabel))

# import matplotlib.image as mpimg
# imgplot = plt.imshow(samples[152, :].reshape(64, 64))
# plt.show()

_mean = np.mean(samples, axis=1).reshape(-1, 1)
variance = (np.std(samples, axis=1)**2).reshape(-1, 1)
samples = (samples - _mean)/variance

samples_sparse = coo_matrix(samples)
samples, samples_sparse, labels = shuffle(samples, samples_sparse, labels)

train_data, test_data, train_label, test_label = train_test_split(samples, labels, test_size=0.30, random_state=4)


# In[ ]:


print(samples.shape, labels.shape)


# ## train section

# In[ ]:


import time

tic = time.time()
activation_function = 'tanh'
model = multi_Layered_NN(train_data.T,
                         train_label,
                         activation=activation_function,
                         n_hidden=[100, 200, 200, 100],
                         num_iterations=10000,
                         learning_rate=0.01,
                         _lambda = 0.000000001,
                         keep_prob_dropout= 0.7,
                         print_cost=True)

print("Train phase time:", time.time()-tic)


# In[ ]:


pred_labels = predict(train_data, parameters=model['parameters'], activation=activation_function)
print('accuracy on train set:', (np.sum(pred_labels == train_label)/pred_labels.size) * 100, '%')
pred_labels = predict(test_data, parameters=model['parameters'], activation=activation_function)
print('accuracy on test set:', (np.sum(pred_labels == test_label)/pred_labels.size) * 100, '%')


# ### plot cost value

# In[ ]:


plt.plot(range(len(model['cost_history'])), model['cost_history'])
plt.show()


# ### test an example of test set

# In[ ]:


photo_number = 11
imgplot = plt.imshow(test_data[photo_number].reshape(64, 64))
plt.show()
pred_labels = predict(test_data[photo_number].reshape(1, -1), parameters=model['parameters'], activation=activation_function)
print(pred_labels)
if int(pred_labels[0][0]) == 1:
    print('Human')
else:
    print('Not-Human')


# ### predict an example from you
# It must be a 64*64 photo.

# In[ ]:


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

    pred_labels = predict(myimg.reshape(1, -1), parameters=model['parameters'], activation=activation_function)
    if int(pred_labels[0][0]) == 1:
        print('Human')
    else:
        print('Not-Human')


# In[ ]:


m = 6
plt.subplot(m,1, 1)
predict_new_photo('drive/ColabNotebooks/myimg.jpg')
plt.subplot(m,1, 2)
predict_new_photo('drive/ColabNotebooks/myimg2.jpg')
plt.subplot(m,1, 3)
predict_new_photo('drive/ColabNotebooks/myimg3.jpg')
plt.subplot(m,1, 4)
predict_new_photo('drive/ColabNotebooks/myimg4.jpg')
plt.subplot(m,1, 5)
predict_new_photo('drive/ColabNotebooks/myimg6.jpg')
plt.subplot(m,1, 6)
predict_new_photo('drive/ColabNotebooks/myimg5.jpg')


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

