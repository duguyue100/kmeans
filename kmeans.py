"""
This file contains my implementation of the paper:
Learning Feature Representation with K-means

Author: Yuhuang Hu
Email: duguyue100@gmail.com
"""

import numpy as np;
import matplotlib;
matplotlib.use('tkagg');
import matplotlib.pyplot as plt;

import util;

# Read image

## MNIST
# train_set, valid_set, test_set=util.load_mnist("mnist.pkl.gz");  
# X=util.sample_patches_mnist(train_set, 5000, 16);

### CIFAR-10
data_set=util.load_CIFAR_batch("./cifar-10/data_batch_1");
data_x=data_set[0]/255.0;
data_x=np.mean(data_x, axis=3);

X=util.sample_patches_cifar10(data_x, 5000, 16);

# Normalization and whitening

X=util.normalize_data(X);
print "[MESSAGE] Data is normalized"

X=util.ZCA_whitening(X);
print "[MESSAGE] Data is whitened"

# plt.figure(1);
# for i in xrange(100):
#   plt.subplot(10,10,i+1);
#   plt.imshow(X[:,i].reshape(16,16), cmap = plt.get_cmap('gray'), interpolation='nearest');
#   plt.axis('off')
#       
# plt.show();

# K-means procedure

num_centroids=100;

D=np.random.normal(size=(256, num_centroids));
D=D/np.sqrt(np.sum(D**2, axis=0));

D=util.kmeans(X, D, 20);
  
plt.figure(1);
for i in xrange(100):
  plt.subplot(10,10,i+1);
  plt.imshow(D[:,i].reshape(16,16), cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off');
    
plt.show();
