import jax
from flax import nnx
from functools import partial
from typing import Optional


class Cifar10JAXClassifier(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, num_classes: int) -> None:
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=32,
            kernel_size=(3, 3),
            rngs=rngs,
            padding="SAME",
        )
        self.pool1 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.bn1 = nnx.BatchNorm(num_features=32, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            rngs=rngs,
            padding="SAME",
        )
        self.pool2 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.bn2 = nnx.BatchNorm(num_features=64, rngs=rngs)

        self.fc1 = nnx.Linear(in_features=64 * 8 * 8, out_features=512, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
        self.fc2 = nnx.Linear(in_features=512, out_features=num_classes, rngs=rngs)
        self.num_classes = num_classes

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        x = self.bn1(self.pool1(nnx.relu(self.conv1(x))))
        x = self.bn2(self.pool2(nnx.relu(self.conv2(x))))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.fc1(x))
        x = self.dropout(x, rngs=rngs)
        x = self.fc2(x)
        return x
