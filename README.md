# Pneumonia-Detection-Using-CNN
Pneumonia Detection from Chest X-ray images using Convolutional Neural Networks (CNNs) Used Python


# Abstract
This project presents a pneumonia detection system using CNNs, with an accuracy rate of 88.14 percent on the test dataset. The
system uses the Keras deep learning library, along with OpenCV, NumPy, imageio, and matplotlib, to build and train the CNN
model. The system processes and augments the training data, generates the training and testing datasets, and converts the testing
data into a format that can be used by the CNN. The final CNN model consists of two convolutional layers, followed by max
pooling and batch normalization layers, a flatten layer, and two fully connected layers, with ReLU activation for the hidden layers
and sigmoid activation for the output layer. The system achieved an accuracy rate of 88.14 percent, a precision rate of 86.40 percent,
a recall rate of 96.15 percent, and an F1-score of 91.01 percent.
In summary, the system developed in this project presents an effective solution for pneumonia detection using CNNs. The system
achieved a high accuracy rate on the test dataset, demonstrating its ability to accurately classify chest X-ray images as normal or
showing signs of pneumonia. The use of data augmentation and CNNs allowed for robust performance in the face of varied input
data. The system’s precision, recall, and F1-score metrics indicate that it can accurately identify positive cases of pneumonia while
minimizing the number of false positives. The system can be further improved by increasing the dataset size and implementing
more complex architectures of CNN models.


### Keywords: pneumonia detection, convolutional neural networks, deep learning, Keras, OpenCV, image processing, medical image analysis, python


# Introduction
Pneumonia is a serious infectious disease that affects the lungs and can lead to severe health complications if not diagnosed and treated in time. The traditional approach to pneumonia detection involves chest X-rays and clinical examinations, which can be time-consuming and require skilled healthcare professionals. To address this issue, we have developed a pneumonia detection system using convolutional neural networks (CNNs) with the Keras deep learning library with a high accuracy rate of 88.14 percent. Our system can accurately detect pneumonia from chest X-ray images, making it a valuable tool for early diagnosis and treatment. The system utilizes various supporting libraries, such as OpenCV, NumPy, imageio, and matplotlib, to process the X-ray images and generate training and testing datasets. The CNN model built using Keras consists of several layers, including convolutional, pooling, batch normalization, and fully connected layers. The activation and loss functions used are ReLU and sigmoid, respectively, to ensure efficient binary classification

# Literature Review
Pneumonia detection has become a significant research area due to its severity and high prevalence in the world. Various techniques have been proposed for pneumonia detection, including feature-based and deep learning-based approaches.
In the study by Varshni et al. (2020) [12], a CNN-based feature extraction method was used for pneumonia detection, achieving an accuracy of 90.67%. In comparison, Chatterjee et al. (2021) [3] proposed an efficient pneumonia detection system with a pre-trained deep learning model on a chest X-ray dataset, achieving an accuracy of 95.68%. Both [12, 3] studies focused on improving the accuracy of pneumonia detection using CNN-based methods. 
On the other hand, Kaswidjanti et al. (2020) [6] proposed a content-based image retrieval method using the Gray Level Co-Occurrence Matrix for pneumonia detection, achieving an accuracy of 80%. DeePNeu by Ordiyasa and Bintang (2020) [8] used Faster R-CNN for robust pneumonia symptom detection and reported an accuracy of 86.58%. Kaushik et al. (2020) [10] proposed a CNN-based approach for pneumonia detection and reported an accuracy of 88.14%.
Overall, these studies demonstrate the effectiveness of deep learning-based approaches for pneumonia detection, with accuracy rates ranging from 80% to 95.68%. However, the reviewed papers [12, 3, 6, 8, 10] also highlight the limitations of CNN-based pneumonia detection systems, such as the need for large, annotated datasets, the potential for bias and overfitting, and the lack of interpretability of the deep learning models. The proposed pneumonia detection system using CNNs should consider these challenges to improve the accuracy of the system while reducing false negatives and false positives.

# Dataset
In this paper, we are using the publically available Chest X-ray images (Pneumonia) dataset on Kaggle [11]. The used dataset consists of 5216 training images, 624 testing images, and 16 validation images. Each set is further divided into two categories: normal and pneumonia. Fig.1
