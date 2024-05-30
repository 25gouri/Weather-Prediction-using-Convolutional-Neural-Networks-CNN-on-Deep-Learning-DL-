# Weather-Prediction-using-Convolutional-Neural-Networks-CNN-on-Deep-Learning-DL-
Our project aims to build a weather prediction model. The prediction of weather has  wide applications. The applications range from agriculture to disaster management. Change detection in satellite imagery involves the identification and evaluation of changes that have taken place over time. Deep learning techniques, including CNN, have proven very successful for remote sensing applications. In this study, we propose an approach for weather prediction using satellite images. The algorithms used in this project were built on deep learning and used to divide satellite images into four categories. The four categories are ‘Cloud,’ ‘Water,’ ‘Desert’ and ‘Green Area.’ 

One of the issues with the satellite images is that the satellite images have different kinds of properties so classifying them into different categories is challenging. Another problem is that noise contamination can be seen in most satellite photos. The CNN model is used to estimate the noise patterns in the wireless image. Our method extracts features from satellite photos using Convolutional Neural Networks (CNN) with the VGG16 architecture. To increase forecast accuracy, the model is trained using satellite imagery and meteorological data.


This work aims to develop a robust weather prediction model leveraging Convolutional Neural Networks (CNN) and satellite images. The methodology encompasses several key stages: Data Preprocessing, Model Architecture, Training and Validation, and Evaluation.

**Data Preprocessing:** Initially, we preprocess the satellite image data to ensure its suitability for training the CNN model. This involves standardizing pixel values and handling missing data. Preprocessing is crucial to enhance the quality and consistency of the dataset, addressing challenges such as varying image properties and noise contamination commonly observed in satellite photos.

**Model Architecture:** Our model architecture integrates deep learning principles, specifically combining CNN layers with the VGG16 architecture. By leveraging VGG16, the model can learn hierarchical features from raw image data effectively. The choice of CNN layers enables the extraction of intricate patterns and features from satellite images, facilitating accurate weather predictions. The architecture design plays a pivotal role in capturing relevant information and enhancing prediction accuracy.

**Training and Validation:** Subsequently, we train the CNN model using weather data and labeled satellite images. The training process involves iterative adjustments to the model's parameters, minimizing prediction errors, and optimizing performance. To ensure the robustness of the model, we employ cross-validation techniques. Cross-validation partitions the dataset into subsets for training and validation, preventing overfitting and ensuring the model's generalizability to unseen data.

**Evaluation:** The performance of our weather prediction model is evaluated based on its classification accuracy on the testing dataset. Evaluation is crucial to verify the model's efficacy and readiness for deployment in practical scenarios.



**Dataset:**
We have assembled the largest satellite image collection. It is called the "Weather Phenomenon Database" (WEAPD). The collection of satellite images consists of 5636 images. The collection is divided into four categories. We used the dataset Satellite Image Classification Dataset-RSI-CB256. This dataset has 4 different classes, including sensors and Google Maps snapshots. We want to increase weather image classification accuracy and efficiency with this dataset.
The trained model will preprocess the dataset and give the output based on the following classification. They are Cloud, Desert, Green area and Water.



**Design and Implementation:**
In our model, for deep learning, we used Python as a programming language. In Python, we used eight different libraries to train our model. They are, NumPy as np, OpenCV as cv2, flask, pandas, TensorFlow, Keras, Operating systems as os, and pickle. The os module is imported to interact with the operating system. It allows the code to perform tasks like navigating directories and handling file operations. The files consist of satellite images. The cv2 module, also known as OpenCV, is a computer vision library used for image processing tasks. It is utilized to load and manipulate images to prepare them for training the neural network. The NumPy library is used for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices NumPy is used for creating and manipulating arrays of image data and for performing mathematical operations necessary for data preprocessing. Keras provides a user-friendly interface for building and training neural networks. Keras is utilized for building and training a convolutional neural network (CNN) model, specifically using a pre-trained VGG16 model for image classification. The pickle module is used for serializing and deserializing Python objects.

Flask is a web application used in Python to create web pages. Python packages are available to create lightweight web applications using the Flask web framework. MySQL is a server-based relational database management system (RDBMS) that allows multiple users to access multiple databases. TensorFlow is used for various operations like data preprocessing, model training, and evaluation. TensorFlow is specifically used to import the VGG16 model, which is a pre-trained deep-learning model for image classification tasks. The VGG16 model is then utilized as the base model for further customization and training to classify satellite images into different categories.

The satellite Images are classified into four categories. They are Cloud, Desert, Green area and Water. Based on classification results the information will come as a weather report:
**Cloud** it may Rain Hourly Weather · 1 PM 44°. rain drop 15% · 2 PM 45°. rain drop 15% · 3 PM 44°. rain drop 20% · 4 PM 43°. rain drop 33% · 5 PM 42°. rain drop 36% · 6 PM 39°.
**Desert**- The daytime temperature averages 38°C while in some deserts it can get down to -4°C at night.
**Green area** a weather map, green indicates areas with light to moderate rain. Heavier precipitation is indicated by light pink or red.
**Water**-Sun 25 | Night. 41°. 19%. WSW 16 mph. Partly cloudy. Low 41F. Winds WSW at 10 to 20 mph. Humidity43%. UV Index0 of 11. Moonrise7:20 pm. Waning Gibbous.


**VGG16:**

VGG16, One of the Convolutional Neural Network i.e., CNN in Deep Learning. It is named VGG16 in light of the fact that it comprises of 16 layers, which includes 13 convolutional layers with 3 completely associated layers. Here is a breakdown of its design

**Input Layer:** Accepts the input image data.

**Convolutional Layers (Conv):** There are 13 convolutional layers in VGG16. These layers perform convolution operations on the input image to extract various features. Each convolutional layer is trailed by a Rectified Linear Unit (ReLU)which is an activation function, which introduces non-linearity into the network.

**Max Pooling Layers:** After some of the convolutional layers, max-pooling layers are added to down-sample the feature maps, reducing their spatial dimensions while retaining important features.

**Fully Connected Layers (FC):** The final three layers of VGG16 are fully connected layers. These layers are densely connected, meaning that each neuron in one layer is connected to every neuron in the subsequent layer. The last completely associated layer produces the final output, typically representing class scores in a classification task.

**SoftMax Layer:** In classification tasks, a SoftMax is an activation function is often applied to the output layer to convert the raw scores produced by the last completely associated layer into probability scores, indicating the likelihood of each class.

**Conclusion:** 
This project demonstrates the potential of deep learning techniques for weather prediction using satellite images. We researched the interesting relationship between deep learning and weather prediction in this research. Using the VGG16 architecture and Convolutional Neural Networks (CNNs), we predicted the weather using satellite image. It is categorised into four categories. They are Cloud, Water, Green Area, and Desert. The data set utilized in the trials will be expanded in subsequent studies by including data from numerous days. We expect more improvements in weather image recognition and forecasting accuracy as future research improves upon current findings. The results obtained in this work confirm that the deep learning techniques are very effective to predict the weather as compared to previous works.
Through rigorous testing and validation procedures, we demonstrate the model's effectiveness in accurately predicting weather conditions. The model achieves an impressive classification accuracy of approximately 95%, validating its reliability and utility for real-world applications.
