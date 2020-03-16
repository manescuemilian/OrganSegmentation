from unet import unet
from preprocess import X_val1, X_val2, Y_val, X_train1, X_train2, Y_train
from preprocess import im_width, im_height
# from augment import train_generator, val_generator
import numpy as np

model = unet((im_height, im_width, 1))

# Data Augmentation (Commented because of lack of computational power)
'''
training_number = 50
validation_number = 20

for i in range(training_number):
    [X1, X2], Yi = next(train_generator)
    X_train1 = np.append(X_train1, [X1], axis = 0)
    X_train2 = np.append(X_train2, [X2], axis = 0)
    Y_train = np.append(Y_train, [Yi], axis = 0)
    
for i in range(validation_number):
    ([X1, X2], Yi) = next(val_generator)
    X_val1 = np.append(X_val1, [X1], axis = 0)
    X_val2 = np.append(X_val2, [X2], axis = 0)
    Y_val = np.append(Y_val, [Yi], axis = 0)
'''

results = model.fit([X_train1, X_train2], Y_train, 
                    validation_data = ([X_val1, X_val2], Y_val),
                    epochs = 100, verbose = 1)

# Saving the model
model.save_weights("model.h5")
print("Saved model to disk")