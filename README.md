# Uniform Noise layer for Keras
This is a simple Keras layer to add random noise drawn from a uniform distribution to a Tensor.

# Usage:
```python
import keras
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input
from uniform_noise import UniformNoise

inputs = Input((10, 10, 3))
conv1 = Conv2D(4, (2,2), activation='relu')(inputs)
conv1 = UniformNoise(minval=-5.0, maxval=-5.0)(conv1)
conv2 = Conv2D(2, (2,2), activation='relu')(conv1)
conv2 = UniformNoise()(conv2)  # Will use default minval=-1.0 and maxval=1.0
fc = Flatten()(conv2)
out = Dense(5, activation='softmax')(fc)

model = Model(inputs=inputs, outputs=out)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# For some given x and y
model.fit(x, y)
```
