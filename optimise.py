import sys
from unet import unet
from preprocess import input_shape, im_height, im_width
import numpy as np
import matplotlib.pyplot as plt

scan_path = sys.argv[1]
seg_path = sys.argv[2]

# Loading the trianed model
model = unet(input_shape)
model.load_weights("model.h5")

X1 = np.zeros((1, im_height, im_width, 1))
X2 = np.zeros((1, im_height, im_width, 1))

# Read the scan file
with open(scan_path, 'r') as f:
        matrix = [[int(num) for num in line.split(' ')] for line in f]
        matrix = np.array(matrix)
        matrix = matrix[:, :, np.newaxis]
        X1[0] = matrix
        
# Read the seg file
with open(seg_path, 'r') as f:
        matrix = [[int(num) for num in line.split(' ')] for line in f]
        matrix = np.array(matrix)
        matrix = matrix[:, :, np.newaxis]
        X2[0] = matrix
        
# Get prediction and write to output file
prediction = model.predict([X1, X2])        
        
output_file = "optim.out"
prediction = prediction[0, :, :, 0]
np.savetxt(output_file, prediction)

# Plot the mask segmentation obtained
fig, ax = plt.subplots()
ax.imshow(prediction, cmap = "gray")