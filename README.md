# Arrhythmia-classification-using-1-D-ResNet

## Introduction

This project endeavors to elegantly harness the power of deep learning model - ResNet in the classification of arrhythmia, using electrocardiogram (ECG) signals. By juxtaposing traditional machine learning methods such as K-nearest neighbor, decision tree, and logistic regression with cutting-edge deep learning algorithms like CNN, LSTM, ResNet, and GRU, it seeks to showcase the superior adaptability of the latter. Drawing upon ECG datasets from Kaggle, this project aspires to create a singular architecture for arrhythmia classification. Moreover, it aims to demonstrate how these trained deep learning models can serve as an inference system, offering valuable insights and predictions for new data in the realm of heart health.

## Dataset
### Description
The Kaggle arrhythmia ECG heartbeat categorization database, a derivative of the MIT-BIH Arrhythmia dataset, comprises 109,446 ECG beats with 187 features. It is resampled at 125 Hz and categorizes heartbeats into five classes: N (Normal Beat), S (Supraventricular premature beat), V (Premature ventricular contraction), F (Fusion of ventricular and normal beat), and Q (Unclassifiable beat). This dataset differs from the PTB dataset, which uses 1 for abnormal and 0 for normal heartbeats.

### Pre-Processsing
Data pre-processing techniques play a major role in developing an accurate prediction model since real-world data is subject to incompleteness, noise, and inconsistency due to its numerous data sources. When a deep learning and machine learning classifier have trained on an incomplete dataset, it outputs incorrect predictions; hence, data pre-processing techniques are required to improve the classification model's training and prediction outputs. The arrhythmia classification model includes the following data pre-processing steps: exploratory data analysis to better understand the dataset's properties and Resampling.

## Model Description
Residual Neural Network (ResNet) is a deep neural network architecture that enables the training of deep networks by introducing residual connections, which bypass some of the layers and enable the learning of residual mapping between the input and the output. This approach helps to mitigate the vanishing gradient problem and allows the network to be deeper without being prone to overfitting or degradation in performance. ResNet has achieved state-of-the-art performance on a wide range of computer vision tasks, such as image classification, object detection, and segmentation.

ResNet uses the ReLU activation function in the residual blocks, which is computationally efficient and has been shown to work in deep neural networks. Batch normalization is a method for normalizing the inputs to a layer in a network, which can help to improve the train-ing process and prevent overfitting. The 3x3 convolutional layer is used in ResNet to learn a hierarchy of features at multiple scales, and can reduce the spatial dimensions of the input feature map while increasing the depth or number of channels. These architectural details are important for understanding the circuit structure of ResNet.

## Performance
A ResNet (Residual Network) model was trained with remarkable success, achieving a training accuracy of 99.90% and a testing accuracy of 99.94%. These high accuracy values indicate that the model performed exceptionally well in classifying the data. Additionally, the training and testing loss values were relatively low, with a training loss of 0.0576 and a testing loss of 0.046. These low loss values suggest that the model's predictions were very close to the actual target values, signifying a strong predictive performance.

## Conclusion
The ResNet model demonstrated exceptional classification capabilities with near-perfect accuracy on both the training and testing datasets. The low training and testing loss values further validate the model's predictive accuracy and generalization to unseen data. These results indicate that the ResNet architecture is a powerful tool for solving classification problems and showcases the effectiveness of deep learning models in this context. It's crucial to consider the context and domain of the classification task to fully appreciate the significance of these outstanding results, as they may vary in real-world applications.


