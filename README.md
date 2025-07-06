🗑️ Garbage Classification using Transfer Learning
This project classifies different types of garbage—such as plastic, metal, paper, and more—using a deep learning model built with Transfer Learning (MobileNetV2). It aims to promote waste segregation at the source, which is vital for efficient recycling and waste management.

📌 Table of Contents
Overview
Tech Stack
Dataset
Model Architecture
How to Run
Prediction on New Image
Results
Conclusion

📖 Overview
The goal of this project is to build an image classification model that can automatically detect the type of garbage from an image. This can help automate the sorting process in smart cities and recycling plants, leading to more sustainable waste handling.

🔧 Tech Stack
Python
TensorFlow / Keras
MobileNetV2 (Pre-trained on ImageNet)
Google Colab / Jupyter Notebook
Matplotlib

📁 Dataset
The dataset used is trashtype_image_dataset, containing images categorized into the following folders:
cardboard/
glass/
metal/
paper/
plastic/
trash/
The dataset is split automatically (80% train, 20% validation) using Keras' ImageDataGenerator.
link to download dataset: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset

🧠 Model Architecture
Base Model: MobileNetV2 (with pre-trained ImageNet weights)
Custom Layers Added:
GlobalAveragePooling2D()
Dense(128, activation='relu')
Dropout(0.3)
Dense(num_classes, activation='softmax')

The model is lightweight and optimized for fast inference.

▶️ How to Run
Clone or download this repository
Structure your dataset as:
trashtype_image_dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
Open and run Garbage_Classification.ipynb
Train the model and observe training/validation accuracy and loss
Upload a test image to see predictions

🖼️ Prediction on New Image
The notebook includes functionality to:
Upload a new test image (e.g., plastic.jpg)
Predict its class using the trained model
Display the image along with the predicted label

📊 Results
Achieved good validation accuracy using a lightweight MobileNetV2 model
Accuracy and loss graphs are plotted to monitor training progress
Performs well with minimal training time

✅ Conclusion
This project demonstrates that transfer learning with MobileNetV2 can be effectively used for real-time garbage classification. The model is compact, accurate, and can be deployed in:
Smart bins
Recycling units
Mobile applications
for improving sustainability and waste management efficiency.
