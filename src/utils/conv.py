import numpy as np

def conv2d(image, kernel, stride=1):
    kernel_size = kernel.shape[0]
    output_size = (image.shape[0] - kernel_size) // stride + 1
    output = np.zeros((output_size, output_size))
    
    for y in range(0, output_size):
        for x in range(0, output_size):
            region = image[y:y+kernel_size, x:x+kernel_size]
            output[y, x] = np.sum(region * kernel)
    
    return output

def relu(feature_map):
    return np.maximum(0, feature_map)

def max_pooling(feature_map, size=2, stride=2):
    h, w = feature_map.shape
    output_h = (h - size) // stride + 1
    output_w = (w - size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    for y in range(output_h):
        for x in range(output_w):
            region = feature_map[y*stride:y*stride+size, x*stride:x*stride+size]
            output[y, x] = np.max(region)
    
    return output

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9)
