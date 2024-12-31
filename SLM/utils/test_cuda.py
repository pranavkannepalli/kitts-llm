import tensorflow as tf
import torch

tf.debugging.set_log_device_placement(True)

# Example tensor operation to check device placement
x = tf.random.normal([3, 3])
y = tf.random.normal([3, 3])
z = tf.matmul(x, y)
print(z)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Set to 0 for full logging
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

print("TensorFlow Version:", tf.__version__)
print("CUDA Built with TensorFlow:", tf.test.is_built_with_cuda())
