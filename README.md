# Big-Data-Application-for-Network-Level-Travel-Time-Prediction With Hierarchical LSTM

-----
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/features-of-spark.jpg" /></p>

Highlights
----------
* Big data tools, such as Apache Spark and Apache MXNet, were applied in travel time processing, modeling, and prediction, which are scalable computing platforms designed for big data workloads. 
* The Autocorrelation analysis and stationary transformation were performed to reveal the underlying properties of travel time. We analyzed a four-year travel time dataset, which is by far the biggest from the Caltrans Performance Measurement System (PeMS) for District 4 Bay Area (from August-1st-2017 to Oct-31st-2021), at a 5-min time frequency. 
* Existing travel time prediction models with LSTM as the backbone are built on stacked architecture without hierarchical feature extraction capability. We proposed a hierarchical pooling module to capture information of different time steps, which is analogous to the human visual perception system that summarizes low-level input into high-level abstractions.
* The self-attention is incorporated that passes on extracted features from the LSTM layers to fully connected layers to add robustness and accuracy. 



Abstract
--------
As a key input for trip planning and congestion management, predicted travel time is essential to the advanced traveler information systems (ATIS). The travel time data collected from widespread traffic monitoring sensors necessitate big data analytic tools for querying, visualization, and identifying meaningful traffic patterns. This paper utilizes a large-scale travel time dataset from Caltrans Performance Measurement System (PeMS) system that is an overflow for traditional data processing and modeling tools. To overcome the challenges of the massive amount of data, the big data analytic engines Apache Spark and Apache MXNet are applied for data wrangling and modeling. Stationarity and autocorrelation were performed to explore and visualize the trend of time-varying data. Inspired by the success of the hierarchical architecture for many Artificial Intelligent (AI) tasks, we consolidate the cell and hidden states passed from low-level to high-level LSTM with an attention pooling similar to how the human perception system operates. The designed hierarchical LSTM model can consider the dependencies at different time scales to capture the spatial-temporal correlations of network-level travel time. Another self-attention module is then devised to connect LSTM extracted features to the fully connected layers, predicting travel time for all corridors instead of a single link/route. The comparison results show that the Hierarchical LSTM with Attention (HierLSTMat) model gives the best prediction results at 30-minute and 45-min horizons and can successfully forecast unusual congestion. The efficiency gained from big data analytic tools was evaluated by comparing them with popular data science and deep learning frameworks.

## Attention Pooling
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/Attention%20Pooling.drawio%20(2).png" /></p>

## Hierarchical LSTM with Attention
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/HierLSTMat.png" /></p>

## One Week Travel Time Prediction Sample
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/Predictions_Comparison.png" /></p>

## 1000 Epochs Training Time for Tensorflow, Pytorch and MXNet
In our experiments, the MXNet has the fastest training speed to finish 1000 epochs after 891.01 seconds, which is approximately 25% faster than Pytorch and 50% faster than Tensorflow.
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/Speed%20Testing.png" /></p>


License
-------
The source code is available only for academic/research purposes (non-commercial).


Contributing
--------
If you found any issues in our model or new dataset please contact: terry.tianya.zhang@gmail.com

