import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import warnings
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

REMOVABLE_PATH = 'COVID-19_Radiography_Dataset/'
PATH = 'COVID-19_Radiography_Dataset/COVID/COVID-54.png'
categories_dict = {0: "Normal", 1: "Viral Pheumonia", 2: "COVID"}
categories = ['Normal', 'Viral Pheumonia', 'COVID']

category = PATH.split('/')[1].split('/')[0]
if category in categories_dict.values():
    print(f'True value is : {category}')

image = load_img(PATH, target_size=(224, 224))
img = img_to_array(image)
img = img.reshape((1, 224, 224, 3))

model = keras.models.load_model('covid.h5')

result = model.predict(img)
result_index = np.argmax(result, axis=-1)
print('Prediction is:', categories_dict[int(result_index)])

result_new = []
for i in range(len(result[0])):
    result_new.append(result[0][i] * 100)

plt.bar(categories, result_new)
plt.show()
plt.imshow(image)
plt.show()
