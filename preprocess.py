import numpy as np
import os

input_shape = (512, 512, 1)
im_width = input_shape[0]
im_height = input_shape[1]

image_path = "./dataset"

input_paths = ["input/107-HU.in", "input2/89-HU.in", "input3/121-HU.in", "input4/187-HU.in"]
seg_paths = ["input/107-seg.in", "input2/89-seg.in", "input3/121-seg.in", "input4/187-seg.in"]
opt_paths = ["input/107-opt.out", "input2/89-opt.out", "input3/121-opt.out", "input4/187-opt.out"]

number_files = len(input_paths)

X_scan = np.zeros((number_files, im_height, im_width, 1))
X_seg = np.zeros((number_files, im_height, im_width, 1))
Y = np.zeros((number_files, im_height, im_width, 1))

for i, file_in in enumerate(input_paths):
    in_img = os.path.join(image_path, file_in)
    with open(in_img, 'r') as f:
        matrix = [[int(num) for num in line.split(' ')] for line in f]
        matrix = np.array(matrix)
        matrix = matrix[:, :, np.newaxis]
        X_scan[i] = matrix
        

for i, file_seg in enumerate(seg_paths):
    seg_img = os.path.join(image_path, file_seg)
    with open(seg_img, 'r') as f:
        matrix = [[int(num) for num in line.split(' ')] for line in f]
        matrix = np.array(matrix)
        matrix = matrix[:, :, np.newaxis]
        X_seg[i] = matrix
        
for i, file_opt in enumerate(opt_paths):
    opt_img = os.path.join(image_path, file_opt)
    with open(opt_img, 'r') as f:
        matrix = [[int(num) for num in line.split(' ')] for line in f]
        matrix = np.array(matrix)
        matrix = matrix[:, :, np.newaxis]
        Y[i] = matrix 

indices = np.arange(number_files)
np.random.shuffle(indices)
X_train1 = X_scan[:3]
X_val1 = X_scan[np.newaxis, 3]
X_train2 = X_seg[:3]
X_val2 = X_seg[np.newaxis, 3]

Y_train = Y[:3]
Y_val = Y[np.newaxis, 3]


print("X train scan shape is " + str(X_train1.shape))
print("X val scan shape is "+ str(X_val1.shape))
print("X train seg shape is " + str(X_train2.shape))
print("X val seg shape is "+ str(X_val2.shape))
print("\n")

print("Y train shape is " + str(Y_train.shape))
print("Y val shape is "+ str(Y_val.shape))