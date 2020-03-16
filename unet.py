from keras.models import Model, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras import backend as K

smooth = 0.00001
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def unet(input_shape):
    # CNN for doctor segmentation input
    inputs_seg = Input((input_shape))
    cs1 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (inputs_seg)
    cs1 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs1)
    ps1 = MaxPooling2D((2, 2)) (cs1)

    cs2 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (ps1)
    cs2 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs2)
    ps2 = MaxPooling2D((2, 2)) (cs2)

    cs3 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (ps2)
    cs3 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs3)
    ps3 = MaxPooling2D((2, 2)) (cs3)
    
    cs4 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (ps3)
    cs4 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs4)
    
    us5 = Conv2DTranspose(32, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (cs4)
    us5 = concatenate([us5, cs3])
    cs5 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (us5)
    cs5 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs5)

    us6 = Conv2DTranspose(16, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (cs5)
    us6 = concatenate([us6, cs2])
    cs6 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (us6)
    cs6 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs6)

    us7 = Conv2DTranspose(8, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (cs6)
    us7 = concatenate([us7, cs1], axis=3)
    cs7 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (us7)
    cs7 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (cs7)
    
    print (cs7.shape)
    
    # CNN for CT scan input
    inputs_scan = Input((input_shape))
    c1 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (inputs_scan)
    c1 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c9)

    print (c9.shape)
    out_concatenated = concatenate([c9, cs7], axis = -1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (out_concatenated)

    model = Model(inputs=[inputs_scan, inputs_seg], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy', dice_coef])
    model.summary()

    return model