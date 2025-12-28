import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import mlx.core as mx
import mlx.nn as nn

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from my_jax.model import Cifar10JAXClassifier
from my_mlx.model import Cifar10MLXClassifier

import os


def load_data(split, is_train=False):
    ds = tfds.load("cifar10", split=split)
    ds = ds.map(
        lambda s: {"image": tf.cast(s["image"], tf.float32) / 255, "label": s["label"]}
    )
    if is_train:
        ds = ds.shuffle(1024).repeat()
    return ds.batch(128).prefetch(1)


test_ds = load_data("test", is_train=False)

print("Test Shape: ", tf.data.experimental.cardinality(test_ds).numpy())

pwd = os.getcwd()

jax_checkpoint_path = os.path.join(pwd, "my_jax", "checkpoint", "state")
mlx_checkpoint_path = os.path.join(pwd, "my_mlx", "checkpoint", "model.safetensors")

mlx_model = Cifar10MLXClassifier()
mlx_model.load_weights(mlx_checkpoint_path)
print("MLX model checkpoint restored.")

checkpointer = ocp.StandardCheckpointer()
abstract_model = nnx.eval_shape(
    lambda: Cifar10JAXClassifier(
        rngs=nnx.Rngs(44),
        num_classes=10,
    )
)
graphdef, abstract_state = nnx.split(abstract_model)

state_restored = checkpointer.restore(jax_checkpoint_path, abstract_state)
print("JAX model checkpoint restored.")

model = nnx.merge(graphdef, state_restored)


@mx.compile
def predict_mlx(x):
    logits = mlx_model(x)
    return logits


@nnx.jit
def predict_jax(model, x):
    logits = model(x, rngs=nnx.Rngs(44))
    return logits


y_true = []
y_pred_jax = []
y_pred_mlx = []

for batch in test_ds.as_numpy_iterator():
    x = mx.array(batch["image"])

    print("Running inference on batch...", x.shape)

    logits_mlx = predict_mlx(x)
    logits_jax = predict_jax(model, jnp.array(batch["image"]))

    preds_mlx = mx.argmax(logits_mlx, axis=1)
    preds_jax = jnp.argmax(logits_jax, axis=1)

    y_true.append(batch["label"])
    y_pred_jax.append(np.array(preds_jax))
    y_pred_mlx.append(np.array(preds_mlx))

y_true = np.concatenate(y_true)
y_pred_jax = np.concatenate(y_pred_jax)
y_pred_mlx = np.concatenate(y_pred_mlx)

import pandas as pd
from sklearn.metrics import classification_report

jax_class_report = classification_report(y_true, y_pred_jax, output_dict=True)
mlx_class_report = classification_report(y_true, y_pred_mlx, output_dict=True)

df = pd.DataFrame.from_dict(
    {
        "Weighted Avg F1 Score": [
            jax_class_report["weighted avg"]["f1-score"],
            mlx_class_report["weighted avg"]["f1-score"],
        ],
        "Macro Avg F1 Score": [
            jax_class_report["macro avg"]["f1-score"],
            mlx_class_report["macro avg"]["f1-score"],
        ],
        "Precision": [
            jax_class_report["weighted avg"]["precision"],
            mlx_class_report["weighted avg"]["precision"],
        ],
        "Recall": [
            jax_class_report["weighted avg"]["recall"],
            mlx_class_report["weighted avg"]["recall"],
        ],
        "Accuracy": [
            jax_class_report["accuracy"],
            mlx_class_report["accuracy"],
        ],
        "Framework": ["JAX", "MLX"],
    }
)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(
    data=df.melt(id_vars=["Framework"]),
    x="variable",
    y="value",
    hue="Framework",
)
plt.title("JAX vs MLX CIFAR-10 Classification Report Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
