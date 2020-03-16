from keras.preprocessing.image import ImageDataGenerator
from preprocess import X_train1, X_train2, Y_train, X_val1, X_val2, Y_val

data_gen_args = dict(rotation_range = 15,
                   width_shift_range = 0.1,
                   height_shift_range = 0.1,
                   shear_range = 0.01,
                   zoom_range = [0.9, 1.25],
                   horizontal_flip = True,
                   vertical_flip = False,
                   fill_mode = 'reflect',
                   data_format = 'channels_last')   

generator = ImageDataGenerator(**data_gen_args)

def generator_two_inputs(X1, X2, Y):
    genX1 = generator.flow(X1, seed = 10)
    genX2 = generator.flow(X2, seed = 10)
    genY = generator.flow(Y, seed = 10)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        Yi = genY.next()
        yield [X1i[0], X2i[0]], Yi[0] 
    
train_generator = generator_two_inputs(X_train1, X_train2, Y_train)
val_generator = generator_two_inputs(X_val1, X_val2, Y_val)