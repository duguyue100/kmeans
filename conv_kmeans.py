# An attempt of implementing conv kmeans in numpy and scipy
# for prove concept
# Author: Hu Yuhuang
# Email: duguyue100@gmail.COM

import numpy as np;
import numpy.linalg as LA;
import scipy.signal as SL;
from scipy.ndimage import imread;
import matplotlib;
matplotlib.use('tkagg');

import matplotlib.pyplot as plt;

img=imread('Lenna.png')/255.0;
img=np.mean(img, axis=2);
img=(img-np.mean(img))/np.sqrt(np.var(img)+10);
# 
# plt.figure(1);
# plt.imshow(img, cmap = plt.get_cmap('gray'), interpolation='nearest');
# plt.axis('off');
# plt.show();

num_centroids=50;

D=np.random.normal(size=(100, num_centroids));
D=D/np.sqrt(np.sum(D**2, axis=0));

D=D.reshape(10, 10, num_centroids);

for i in xrange(10):
  S=np.zeros((503, 503, num_centroids));
  for j in xrange(num_centroids):
    S[:, :, j]=SL.convolve2d(img, D[:,:,j], mode="valid");
  
  S=S*(S>=np.max(S,axis=2,keepdims=True));
  
  for j in xrange(num_centroids):
    temp=SL.convolve2d(img, S[:, :, j], mode="valid");
    D[:, :, j]=temp+D[:, :, j];
    D[:, :, j]=D[:,:,j]/np.sqrt(np.sum(D[:,:,j]**2));
    
  print "[MESSAGE] Iteration %i is done" % (i);

    
plt.figure(1);
for i in xrange(num_centroids):
  plt.subplot(5,10,i+1);  
  plt.imshow(D[:,:,i], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
    
plt.show();