# Audify - The Audio Classifier

What is Audify?

Audify - An Audio Classification project built by a Aritificial Neural Network model that accuartely categorized audio samples based on thier content.

Why is Audify Unique? How does it help real world?

Sound monitoring -> By accurately classifying urban sounds, such as car horns, sirens, and jackhammers, the audio classification model can be used in real-time sound monitoring applications. It can assist city planners, environmental agencies, and policymakers in understanding noise patterns, identifying areas with excessive noise levels, and implementing measures to mitigate noise pollution.
Public Safety and Security -> Ability to classify audio signals in real time can contribute to public safety and security. For example, the audio classification model can be integrated into surveillance systems to automatically detect and recognize critical sounds like gunshots or alarms.


## Table of Contents

## Dataset Link
[Link to dataset](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

The dataset consist of 8732 audio files  in WAV format, 
the dataset includes 10 low level classes and number of files respectively

![WhatsApp Image 2023-04-30 at 20 06 53](https://user-images.githubusercontent.com/122106180/235359054-ca80aca6-f922-4663-b41a-9d5387528132.jpg)


## Tech Stack
### Programming Language: Python
### Python Libraries used
- Numpy
- Pandas
- Scikit
- Keras
- Tersorflow
- Librosa
- Flask
- Pickle
## Project Description
### Feature Extraction
In this project the feature considered is MFCC (Mel Frequency Cepstral Coefficients), MFCCs are a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear Mel frequency scale. The Mel frequency scale is a perceptual scale of frequency that is based on the human ear's response to sound. The power spectrum of the sound is divided into a number of frequency bands that are equally spaced on the Mel scale, and the energy in each band is then summed and logarithmically compressed. This log-compressed spectrum is then transformed using a Discrete Cosine Transform (DCT) to produce the MFCCs. MFCC is one of the commonly used features that has been
used in a variety of applications especially in voice signal processing such as speaker recognition, voice recognition, and
gender identification

The following code is used to extract MFCC value from each file

![WhatsApp Image 2023-04-30 at 20 14 26](https://user-images.githubusercontent.com/122106180/235359508-ce964475-7ebc-4a31-9332-298ebb2b110e.jpg)

the extracted MFCC values are made into Data Frame

![WhatsApp Image 2023-04-30 at 20 16 36](https://user-images.githubusercontent.com/122106180/235359774-00d53292-e0be-4c61-9e01-c48bbdae0323.jpg)

### Data Splitting
The created data set is split into 70-30 for Training and Testing of the Model

![WhatsApp Image 2023-04-30 at 20 33 29](https://user-images.githubusercontent.com/122106180/235360381-23ae2919-ea6a-4a88-aade-fb7452ac9052.jpg)

The Shape of the Training and Testing dataset

![WhatsApp Image 2023-04-30 at 20 34 15](https://user-images.githubusercontent.com/122106180/235360619-4db5662f-2e2f-487d-96fe-5969641066bb.jpg)

### Deep Neural Network Model

In the project we built fully connected network with an input layer ,2 hidden layers and an output layer using sequential API.
The first layer consists of 100 neurons and takes input of 40 features with dropout of 50% and uses RELU activation function.
The second layer  consists of 200 neurons  with dropout of 50% and uses RELU activation function to extract more complex features than the first layer.
The third layer is a dense layer with 100 neurons and a ReLU activation function, again extracting more complex features.
the last layer is another dense layer with the number of output classes (10) neurons and a softmax activation function.

![WhatsApp Image 2023-04-30 at 20 47 56](https://user-images.githubusercontent.com/122106180/235361081-3d88c2db-1816-4044-a035-46c8ccf61edd.jpg)

### Training the Model

We have trained the model for 100 epoch with the batch size of 50

![WhatsApp Image 2023-05-01 at 12 12 19](https://user-images.githubusercontent.com/122106180/235418198-46b0dcec-f449-40dd-9fbb-af83e9052894.jpg)

### Result

From training and test the model we obtained the accuracy of

![WhatsApp Image 2023-05-01 at 12 20 47](https://user-images.githubusercontent.com/122106180/235419127-4bfa0064-652f-4be2-8c12-9a10295d58f6.jpg)

### Comparison with previous works
The following paper : Salamon, Justin, Christopher Jacoby, and Juan Pablo Bello. "A dataset and taxonomy for urban sound research." Proceedings of the 22nd ACM international conference on Multimedia. 2014.

[Link to paper](https://dl.acm.org/doi/abs/10.1145/2647868.2655045?casa_token=KZ5YCFni-awAAAAA:YhQaGFXGPK7gC9dKi8UAXB2Siyi4duOrxjLGWJ6lahqxqQUsi47m6SG4BlwRbW3PcWGtmSWCVyDVag)

The research paper comes to conclusion with the use of SVM and Random Forest model having high accuracy of approximately 73%, From the Neural Network Architecture we used in the project we got a improved accuracy of 83.81 % and 78.28 % for training and testing respectively.

## Front End of the Project

We used Flask web frame work of python for the front end and pickle module for the loading the model.

![WhatsApp Image 2023-05-01 at 13 17 55](https://user-images.githubusercontent.com/122106180/235424892-afb0c7a4-37f7-4272-b960-751787fc7f0f.jpg)

By uploading the sample audio file, the model classifies the sample and give this result in result page by highlighting predicted class.

![WhatsApp Image 2023-05-01 at 13 20 16](https://user-images.githubusercontent.com/122106180/235425135-f5cd1923-a1b9-4137-b393-c1e3e9f2c9f1.jpg)

## Challenges faced
