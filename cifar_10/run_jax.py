import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from my_jax.model import Cifar10JAXClassifier
import optax
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

tf.random.set_seed(44)

train_steps = 1200
eval_every = 200
batch_size = 32

train_ds: tf.data.Dataset = tfds.load("cifar10", split="train")
test_ds: tf.data.Dataset = tfds.load("cifar10", split="test")

train_ds = train_ds.map(
    lambda sample: {
        "image": tf.cast(sample["image"], tf.float32) / 255,
        "label": sample["label"],
    }
)  # normalize train set
test_ds = test_ds.map(
    lambda sample: {
        "image": tf.cast(sample["image"], tf.float32) / 255,
        "label": sample["label"],
    }
)  # Normalize the test set.

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat().shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

model = Cifar10JAXClassifier(rngs=nnx.Rngs(0), num_classes=10)
nnx.display(model)

learning_rate = 5e-4
momentum = 0.9

optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate, momentum),
    wrt=nnx.Param,
)
metrics = nnx.MultiMetric(
    f1_score=nnx.metrics.Average("f1_score"),
    loss=nnx.metrics.Average("loss"),
)

nnx.display(optimizer)


def loss_fn(model: Cifar10JAXClassifier, rngs: nnx.Rngs, batch: dict) -> tuple:
    logits = model(
        batch["image"],
        rngs=rngs,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch["label"],
    ).mean()
    return loss, logits


@jax.jit
def macro_f1_score(y_true, y_pred, num_classes=10):
    # 1. One-hot encode inputs
    # Shape: [batch_size, num_classes]
    y_true = jax.nn.one_hot(y_true, num_classes)
    y_pred = jax.nn.one_hot(y_pred, num_classes)

    # 2. Calculate TP, FP, FN per class (summing over batch dimension axis=0)

    # TP: Predicted class X and actually class X
    tp = jnp.sum(y_true * y_pred, axis=0)

    # FP: Predicted class X but actually NOT class X
    fp = jnp.sum((1 - y_true) * y_pred, axis=0)

    # FN: Actually class X but predicted NOT class X
    fn = jnp.sum(y_true * (1 - y_pred), axis=0)

    # 3. Calculate Precision & Recall per class

    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # 4. Calculate F1 per class
    f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)

    # 5. Return Macro F1 (average across classes)
    return jnp.mean(f1_per_class)


@nnx.jit
def train_step(
    model: Cifar10JAXClassifier,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch: dict,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, rngs, batch)
    f1_score_value = macro_f1_score(
        y_true=batch["label"],
        y_pred=jax.numpy.argmax(logits, axis=-1),
    )
    metrics.update(
        f1_score=f1_score_value,
        loss=loss,
    )
    optimizer.update(model, grads)


@nnx.jit
def eval_step(
    model: Cifar10JAXClassifier,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch: dict,
):
    loss, logits = loss_fn(model, rngs, batch)
    f1_score_value = macro_f1_score(
        y_true=batch["label"],
        y_pred=jax.numpy.argmax(logits, axis=-1),
    )
    metrics.update(
        f1_score=f1_score_value,
        loss=loss,
    )


metrics_history = {
    "train_loss": [],
    "train_f1_score": [],
    "test_loss": [],
    "test_f1_score": [],
}
rngs = nnx.Rngs(44)

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    model.train()
    train_step(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        rngs=rngs,
        batch=batch,
    )

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in metrics.compute().items():  # Compute the metrics.
            metrics_history[f"train_{metric}"].append(value)  # Record the metrics.
        metrics.reset()  # Reset the metrics for the test set.

        model.eval()
        for test_batch in test_ds.as_numpy_iterator():
            eval_step(
                model=model,
                metrics=metrics,
                rngs=rngs,
                batch=test_batch,
            )

        for metric, value in metrics.compute().items():  # Compute the metrics.
            metrics_history[f"test_{metric}"].append(value)  # Record the metrics.
        metrics.reset()  # Reset the metrics for the next training epoch.

model.eval()  # Switch to evaluation mode.

_, state = nnx.split(model)
pwd = os.getcwd()
checkpoint_path = os.path.join(pwd, "my_jax", "checkpoint")
ckpt_dir = ocp.test_utils.erase_and_create_empty(checkpoint_path)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_dir / "state", state)

print(f"Model checkpoint saved at: {ckpt_dir / 'state'}")


@nnx.jit
def pred_step(model: Cifar10JAXClassifier, batch):
    logits = model(batch["image"])
    return logits.argmax(axis=1)


test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(model, test_batch)

print(classification_report(test_batch["label"], pred, digits=4))

fig_preds, axs = plt.subplots(5, 5, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch["image"][i])
    ax.set_title(f"predicted={pred[i]}, actual={test_batch['label'][i]}")
    ax.axis("off")
plt.show()

fig_losses, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title("Loss")
ax2.set_title("F1 Score")
for dataset in ("train", "test"):
    ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
    ax2.plot(metrics_history[f"{dataset}_f1_score"], label=f"{dataset}_f1_score")
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()
plt.show()
