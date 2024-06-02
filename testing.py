import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

classes = ['Apple','Banana','Grape','Mango','Strwaberry']

new_model  = load_model('model/model_last.h5')

test_image = image.load_img("C:/deep learning dataset/archive (11)/Fruits Classification/train/Banana/Banana (1).jpeg",target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = new_model.predict(test_image)
result1 = result[0]
for i in range(6):
    if result1[i] == 1:
        break
prediction = classes[i]
print(prediction)
