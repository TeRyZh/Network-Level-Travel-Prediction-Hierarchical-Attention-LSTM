# Big-Data-Application-for-Network-Level-Travel-Time-Prediction

-----
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/features-of-spark.jpg" /></p>

Highlights
----------
* Big data tools were applied in the travel time prediction task. The reference and proposed models were developed with Apache Spark and Apache MXNet that are scalable computing platforms designed for big data workloads. 
* Existing travel time prediction models with LSTM as the backbone are built on stacked architecture without hierarchical feature extraction capability. We added hierarchical features analogous to the pooling layer in Convolutional Neural Network to capture information of different time sequence. 
* The self-attention module is incorporated that passes on extracted features from the LSTM layers to fully connected layers to add robustness and accuracy. Autocorrelation analysis and stationary transformation were performed to reveal the underlying properties of travel time.
* We published a new travel time dataset, which is by far the biggest from the Caltrans Performance Measurement System (PeMS) District 4 area (from August-1st-2017 to Oct-31st-2021), at a 5-min time-frequency. 


Abstract
--------
This paper used the big data analytic engines Apache Spark and Apache MXNet for data processing and modeling. The efficiency gain was evaluated by comparing it with popular data science and deep learning frameworks. The hierarchical feature pooling is explored for both between layer and the output layer LSTM (Long-Short-Term-Memory). The designed hierarchical LSTM model can consider the dependencies at a different time scale to capture the spatial-temporal correlations from network-level corridor travel time. A self-attention module is then used to connect temporal and spatial features to the fully connected layers, predicting travel time for all corridors instead of a single link/route. Seasonality and autocorrelation were performed to explore the trend of time-varying data. The case study shows that the Hierarchical LSTM with Attention (hiLSTMat) model gives the best result and outperforms baseline models. The California Bay Area corridor travel time dataset covering four-year periods was published from Caltrans Performance Measurement System (PeMS) system. 

## Hierarchical LSTM
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/hiLSTMS.png" /></p>

## Hierarchical LSTM with Attention
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/hiLSTMat.png" /></p>

## 1000 Epochs Training Time for Tensorflow, Pytorch and MXNet
In our experiments, the MXNet has the fastest training speed to finish 1000 epochs after 891.01 seconds, which is approximately 25% faster than Pytorch and 50% faster than Tensorflow. As a deep learning framework optimized for big data applications, the MXNet shows a different loss function over epochs compared to Tensorflow and Pytorch.
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/Speed%20Testing.png" /></p>


License
-------
The source code is available only for academic/research purposes (non-commercial).


Contributing
--------
If you found any issues in our model or new dataset please contact: terry.tianya.zhang@gmail.com

