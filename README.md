# Capstone-GE-Asset-Tracking

Trajectories are one way to think about how things move, from medical equipment within hospitals to taxis in cities to planes across the globe. Although each of these things move very differently in scale and pattern, their movements can all be represented as a series of x,y points.

For our masters capstone as Data Science students at Columbia University, my team and I worked with a mentor from GE Research to cluster those series of points, those trajectories, by shape, because before deciding how things should move, to minimize costs of transport, cost of inventory, etc. we need to understand how they do move. This repository contains a selection of our work.

## Sitemap
`Autoencoders`: Neural network autoencoder results
 - `LSTM_model_shifted_Tabitha.ipynb`: Compares results of training autoencoders to predict the original trajectory versus the trajectory shifted forward by one
 - `EvaluationDataset_Tabitha.ipynb`: Results of best performing model (as trained on simulated data) on another dataset
  
`DistanceMetrics`: Clustering algorithms and distance metrics designed for spaciotemporal data
- `clustereditdist.py`: Functions for distance metrics and clustering algorithm (dbscan)
- `EditDistance.py`: Functions for edit distance algorithm, designed for spaciotemporal data
- `ClusteringTrajectoriesDemo.ipynb`: Demonstration of clustering data with dbscan
- `Clustering Trajectories with Edit Distance.ipynb`: Results of clustering simulated datasets with edit distance as distance metrics and dbscan as clustering algorithm
  
`Visualization`: Visualization using LineCollection
- `visualizationclusters.py`: Script for visualization using python LineCollection library

`GE Asset Tracking_Final Report.pdf`: Report describing project concept, development process, and results
  
