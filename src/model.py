import numpy as np
from utils.conv import conv2d, relu, max_pooling, softmax, cross_entropy

class SimpleCNN:
    def __init__(self):
        # Init random kernels (3x3)
        self.kernel = np.random.randn(3, 3) * 0.1
        self.fc_weights = np.random.randn(225, 2) * 0.1  # 15x15 output dari pooling
        
    def forward(self, image):
        self.conv_out = conv2d(image, self.kernel)
        self.relu_out = relu(self.conv_out)
        self.pool_out = max_pooling(self.relu_out)
        self.flat = self.pool_out.flatten()
        self.fc_out = np.dot(self.flat, self.fc_weights)
        self.out = softmax(self.fc_out)
        return self.out

    def backward(self, image, label, lr=0.01):
        # Forward pass
        output = self.forward(image)
        loss = cross_entropy(output, label)

        # Grad for fully connected
        d_out = output
        d_out[label] -= 1  # dL/dy_pred

        d_fc_weights = np.outer(self.flat, d_out)
        self.fc_weights -= lr * d_fc_weights

        # Grad untuk kernel dll bisa ditambahkan (cukup rumit tapi bisa)
        return loss
