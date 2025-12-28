import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from my_mlx.model import Cifar10MLXClassifier
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot as plt
from functools import partial
import os
from tqdm import tqdm

# --- 1. SETUP ---
# Hide GPU from TensorFlow so it doesn't hog memory
tf.config.set_visible_devices([], "GPU")

batch_size = 32
num_epochs = 10
learning_rate = 5e-4


# Load Data (Standard TFDS)
def load_data(split, is_train=False):
    ds = tfds.load("cifar10", split=split)
    ds = ds.map(
        lambda s: {"image": tf.cast(s["image"], tf.float32) / 255, "label": s["label"]}
    )
    if is_train:
        ds = ds.shuffle(1024).repeat()
    return ds.batch(batch_size).prefetch(1)


train_ds = load_data("train", is_train=True)
test_ds = load_data("test", is_train=False)


# --- 3. HELPER FUNCTIONS ---
def get_f1_components(y_true, y_pred, num_classes=10):
    """Returns TP, FP, FN counts per class"""
    y_true_oh = mx.zeros((y_true.shape[0], num_classes))
    y_true_oh[mx.arange(y_true.shape[0]), y_true] = 1

    y_pred_oh = mx.zeros((y_pred.shape[0], num_classes))
    y_pred_oh[mx.arange(y_pred.shape[0]), y_pred] = 1

    tp = mx.sum(y_true_oh * y_pred_oh, axis=0)
    fp = mx.sum((1 - y_true_oh) * y_pred_oh, axis=0)
    fn = mx.sum(y_true_oh * (1 - y_pred_oh), axis=0)
    return tp, fp, fn


def compute_macro_f1(tp, fp, fn):
    """Calculates F1 from summed components"""
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return mx.mean(f1)  # Macro Average


def loss_fn(model, x, y):
    logits = model(x)
    loss = mx.mean(nn.losses.cross_entropy(logits, y))
    f1_score = compute_macro_f1(*get_f1_components(y, mx.argmax(logits, axis=1)))
    return loss, f1_score


model = Cifar10MLXClassifier()
mx.eval(model.parameters())  # Initialize weights
optimizer = optim.AdamW(learning_rate=learning_rate)

history = {"train_loss": [], "train_f1": [], "test_f1": []}

state = [model.state, optimizer.state]


@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    (loss, f1_score), grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss, f1_score


# Steps per epoch (Since train_ds is infinite)
steps_per_epoch = 1200

print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = []
    running_f1 = []

    train_iter = zip(range(steps_per_epoch), train_ds.as_numpy_iterator())

    for _, batch in tqdm(train_iter, total=steps_per_epoch, desc=f"Epoch {epoch+1}"):
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])

        # 1. Run Step (Lazy)
        loss, f1_score = train_step(x, y)

        # 2. Store Lazy Array
        running_loss.append(loss)
        running_f1.append(f1_score)

    # 3. Batch Eval at end of epoch (Performance Boost)
    mx.eval(running_loss, running_f1)
    epoch_loss = np.mean([l.item() for l in running_loss])
    epoch_f1 = np.mean([f.item() for f in running_f1])
    history["train_loss"].append(epoch_loss)
    history["train_f1"].append(epoch_f1)

    # --- TEST PHASE ---
    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_ds.as_numpy_iterator():
        x = mx.array(batch["image"])
        y = batch["label"]

        logits = model(x)
        preds = mx.argmax(logits, axis=1)

        mx.eval(preds)
        all_preds.append(np.array(preds))
        all_labels.append(y)

    # Concatenate results for Global Metrics
    y_pred_all = np.concatenate(all_preds)
    y_true_all = np.concatenate(all_labels)

    from sklearn.metrics import f1_score

    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    history["test_f1"].append(f1)

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Test F1: {f1:.4f}")

# Save
os.makedirs("checkpoints", exist_ok=True)
model.save_weights("checkpoints/cifar_model.safetensors")
print("Saved!")

test_batch = next(test_ds.as_numpy_iterator())
x_test = mx.array(test_batch["image"])
logits = model(x_test)
preds = mx.argmax(logits, axis=1)

mx.eval(preds)

from sklearn.metrics import classification_report

print(classification_report(test_batch["label"], np.array(preds)))

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(
    history["train_loss"],
    label="Train Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(
    history["train_f1"],
    label="Train F1 Score",
)
plt.plot(
    history["test_f1"],
    label="Test F1 Score",
)  # Plotting acc vs loss is weird, but works for checking
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid()

plt.show()
