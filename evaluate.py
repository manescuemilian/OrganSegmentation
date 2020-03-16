import matplotlib.pyplot as plt
from train import results, model
from preprocess import X_scan, X_seg
import numpy as np

# Plotting the results

# Draw dice coefficient
plt.plot(results.history['dice_coef'])
plt.plot(results.history['val_dice_coef'])
plt.title('Model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

# Draw Loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

# Getting the predictions
predictions = model.predict([X_scan, X_seg])
predictions = predictions[:, :, :, 0] # ditch the channel dimension
print(predictions.shape)

# Writing the results to the output file
for i in range(predictions.shape[0]):
    output_file = "./output/output" + str(i + 1)
    with open(output_file, 'w+') as f:
        matrix_to_write = predictions[i]
        np.savetxt(output_file, matrix_to_write)
        

# Visualize the segmentations
fig, axs = plt.subplots(1, 4, figsize = (15,15))

for i in range(predictions.shape[0]):
    seg_to_show = predictions[i]
    axs[i].imshow(seg_to_show, cmap = "gray")
    
plt.show()