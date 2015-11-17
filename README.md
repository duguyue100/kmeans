# kmeans
A temp project for carrying out convolutional K-means idea.

# Results

+ Configuration:
  + Dataset: MNIST (Used 5k image here)
  + Method: ConvKmeans with Theano
  + Number of Centroids: 100
  + Centroids (filters) size: 28 x 28 (same as MNIST image)
  + Iteration: 10
  + Preprocessing: Normalization, ZCA Whitening
  + Notes: Although I tried with smaller filter size and it shows nice figure, but I guess this can represent it nicely, I chose to use the size as same as image, so it suppose to learn centroids as figures mixed together, and it did.
  
  ![1](https://github.com/duguyue100/kmeans/blob/master/conv_kmeans_theano_100_centroids_28x28_mnist.png "1st result") 