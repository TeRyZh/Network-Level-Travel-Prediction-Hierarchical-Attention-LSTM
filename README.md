# [Network-Level-Traffic-Prediction With Hierarchical Attention LSTM](https://www.maxapress.com/data/article/dts/preview/pdf/dts-0024-0021.pdf)

-----

Highlights
----------
* Introduces a dual attention pooling mechanism that processes both cell states and hidden states across LSTM layers that can better captures complex temporal patterns. 
* More accurate prediction of unusual traffic events and No feature engineering required, unlike graph-based methods. 
* The Autocorrelation analysis and stationary transformation were performed to reveal the underlying properties of travel time. 
*  We proposed a hierarchical pooling module to capture information of different time steps, which is analogous to the human visual perception system that summarizes low-level input into high-level abstractions.
* Outperforms existing models on traffic flow forecasting for PEMSD4 and PEMSD8 datasets Achieves 50% reduction in MAE for PEMSD4 and 38% for PEMSD8 compared to next best models with Efficient model size (1.58MB with 64 hidden states).





Abstract
--------
Traffic state data, such as speed, density, volume and travel time collected from ubiquitous roadway detectors require advanced network level analytics for forecasting and identifying significant traffic patterns. This paper leverages diverse traffic state datasets from the Caltrans Performance Measurement System (PeMS) hosted on the open benchmark and achieved promising performance compared to well recognized spatial-temporal prediction models. Drawing inspiration from the success of hierarchical architectures in various Artificial Intelligence (AI) tasks, we integrate cell and hidden states from low-level to high-level Long Short-Term Memory (LSTM) networks with the attention pooling mechanism, similar to human perception systems. The developed hierarchical structure is designed to account for dependencies across different time scales, capturing the spatial-temporal correlations of network-level traffic states, enabling the prediction of traffic states for all corridors rather than a single link or route. The efficiency of designed hierarchical LSTM is analyzed by ablation study, demonstrating that the attention-pooling mechanism in both cell and hidden states not only provides higher prediction accuracy but also effectively forecasts unusual congestion patterns. Data and code are made publicly available to support reproducible scientific research.

## Attention Pooling
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/Attention%20Pooling.drawio%20(2).png" /></p>

## Hierarchical LSTM with Attention
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/MultiLayHierAttnLSTM.png" /></p>

## One Week Travel Time Prediction Sample
<p align="center"><img src="https://github.com/TeRyZh/Big-Data-Application-for-Network-Level-Travel-Time-Prediction/blob/main/Images/Predictions_Comparison.png" /></p>

## [Code Implementations](https://github.com/TeRyZh/Network-Level-Travel-Prediction-Hierarchical-Attention-LSTM/blob/main/baselines/HierAttnLstm.py)

License
-------
The source code is available only for academic/research purposes (non-commercial).


Contributing
--------
If you found any issues in our model or new dataset please contact: terry.tianya.zhang@gmail.com

