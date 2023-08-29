import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import streamlit as st 
from keras.models import load_model 
from PIL import Image
import json 
import seaborn as sns

model = load_model(r'C:\Users\Aadya Dewangan\Desktop\WoM\model1.h5')

st.title('Crack Detection')
st.text('This CNN model detects cracks from concrete surfaces to montitor the \nhealth of buildings and even roads.')
st.text('It can be connected to a DAQ or programmed inside drones as well,\nand is cost effective')

url1 = 'https://colab.research.google.com/drive/19I-Zgi2QUZmh4V1ezezxMCvt7Fd-u75G?usp=sharing'
st.write("Check out the Model [Here](%s)" % url1)

st.subheader('About the Model')
st.text('Total Dataset: 15000 images of each Positive and Negative Images')
st.text('Dataset used for training:12000 images selected randomly from the entire set,\nbecause it was overfitting')

url2 = 'https://www.kaggle.com/datasets/oluwaseunad/concrete-and-pavement-crack-images'
st.write("[Dataset Source](%s)" % url2)


st.text('No. of CNN layers = 3')
st.text('No.of epochs:12')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png'])


def predict(path):
    test = io.imread(path)
    test_resized = resize(test, (256, 256))
    test_gray = rgb2gray(test_resized)

    test_img = test_gray.reshape((1, 256, 256, 1))
    plt.imshow(test)
    x = model.predict(test_img)[0,0]
    if (round(x) == 1):
        return 'CRACKED'
    else:
        return 'NOT CRACKED'

def plot_confusion_matrix(conf_matr):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    fig = plt.show()
    st.pyplot(fig)



if uploaded_file is not None:
    predicted_class = predict(uploaded_file)
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {predicted_class}")

with open('eval_results.json', 'r') as file:
    res = json.load(file)

    

st.subheader('Evaluation Results')
st.write(f"Test Loss: {res['Test Loss']:.2f}")
st.write(f"Test Accuracy: {res['Test Accuracy']:.2f} %")
st.write(f"R2 Score: {res['R2 Score:']:.2f}")

conf_matrix = np.array(res["Confusion Matrix"])
plot_confusion_matrix(conf_matrix)

st.set_option('deprecation.showPyplotGlobalUse', False)

