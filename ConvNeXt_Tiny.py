import os
import csv
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "convnext_tiny"

DATA_DIR = "dataset_images_v2/train"

BATCH_SIZE = 32
EPOCHS = 15

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01

NUM_CLASSES = 24
IMAGE_SIZE = 224
RANDOM_SEED = 42

NUM_WORKERS = 2

MODEL_DIR = "trained_models"
LOG_DIR = "training_logs"
RESULTS_DIR = "training_results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

run_start = datetime.now()
timestamp = run_start.strftime("%Y-%m-%d_%H-%M-%S")


# ============================================================
# OUTPUT FILES
# ============================================================

txt_log_path = os.path.join(
    LOG_DIR,
    f"{MODEL_NAME}_{timestamp}_training.txt"
)

csv_log_path = os.path.join(
    RESULTS_DIR,
    f"{MODEL_NAME}_{timestamp}_history.csv"
)

checkpoint_path = os.path.join(
    MODEL_DIR,
    f"{MODEL_NAME}_radioml_best.pth"
)

final_checkpoint_path = os.path.join(
    MODEL_DIR,
    f"{MODEL_NAME}_radioml_final.pth"
)


# ============================================================
# LOGGING
# ============================================================

log_file = open(
    txt_log_path,
    "w",
    encoding="utf-8"
)


def log(message=""):
    print(message)
    log_file.write(str(message) + "\n")
    log_file.flush()


log("=" * 70)
log("RadioML ConvNeXt-Tiny Training")
log("=" * 70)

log("")
log(f"Selected model: {MODEL_NAME}")
log(f"Run started: {run_start}")
log(f"Dataset: {DATA_DIR}")
log(f"TXT log: {txt_log_path}")
log(f"CSV history: {csv_log_path}")
log(f"Best checkpoint: {checkpoint_path}")
log(f"Final checkpoint: {final_checkpoint_path}")

log("")
log(f"Random seed: {RANDOM_SEED}")
log(f"Device: {device}")

if torch.cuda.is_available():
    log(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

log("")
log("Training configuration:")
log(f"Batch size: {BATCH_SIZE}")
log(f"Epochs: {EPOCHS}")
log(f"Learning rate: {LEARNING_RATE}")
log("Optimizer: AdamW")
log(f"Weight decay: {WEIGHT_DECAY}")
log(f"Number of classes: {NUM_CLASSES}")
log(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
log("")


# ============================================================
# TRANSFORMS
# ============================================================

data_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

log("Loading dataset...")

full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=data_transform
)

log(f"Total images: {len(full_dataset)}")
log(
    f"Number of classes: "
    f"{len(full_dataset.classes)}"
)
log("")


# ============================================================
# VERIFY NUMBER OF CLASSES
# ============================================================

if len(full_dataset.classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(full_dataset.classes)}."
    )


# ============================================================
# CLASS MAPPING
# ============================================================

log("ImageFolder class mapping:")

for class_name, class_index in (
    full_dataset.class_to_idx.items()
):
    log(
        f"{class_index:2d} -> {class_name}"
    )

log("")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(
    0.8 * len(full_dataset)
)

val_size = (
    len(full_dataset) - train_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

log(
    f"Training images: "
    f"{len(train_dataset)}"
)

log(
    f"Validation images: "
    f"{len(val_dataset)}"
)

log("")


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# INITIALIZE CONVNEXT-TINY
# ============================================================

log("Initializing model: convnext_tiny")
log("")

weights = models.ConvNeXt_Tiny_Weights.DEFAULT

model = models.convnext_tiny(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

model.classifier[2] = nn.Linear(
    model.classifier[2].in_features,
    NUM_CLASSES
)

model = model.to(device)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

log(
    f"Total parameters: "
    f"{total_params:,}"
)

log(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)

log("")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# CSV INITIALIZATION
# ============================================================

with open(
    csv_log_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_time_seconds"
    ])


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_accuracy = 0.0
best_val_loss = float("inf")
best_epoch = 0

training_start_time = time.time()

log("Starting training...")
log("")


# ============================================================
# TRAINING LOOP
# ============================================================

try:

    for epoch in range(EPOCHS):

        epoch_start_time = time.time()

        log("=" * 70)
        log(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        log("=" * 70)


        # ====================================================
        # TRAINING PHASE
        # ====================================================

        model.train()

        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=torch.cuda.is_available()
            )

            labels = labels.to(
                device,
                non_blocking=torch.cuda.is_available()
            )

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate loss
            running_train_loss += (
                loss.item()
                * images.size(0)
            )

            # Predictions
            _, predicted = torch.max(
                outputs,
                1
            )

            train_total += (
                labels.size(0)
            )

            train_correct += (
                predicted == labels
            ).sum().item()


        # ====================================================
        # TRAINING METRICS
        # ====================================================

        train_loss = (
            running_train_loss
            / train_total
        )

        train_accuracy = (
            100.0
            * train_correct
            / train_total
        )


        # ====================================================
        # VALIDATION PHASE
        # ====================================================

        model.eval()

        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=torch.cuda.is_available()
                )

                labels = labels.to(
                    device,
                    non_blocking=torch.cuda.is_available()
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                running_val_loss += (
                    loss.item()
                    * images.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += (
                    labels.size(0)
                )

                val_correct += (
                    predicted == labels
                ).sum().item()


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        val_loss = (
            running_val_loss
            / val_total
        )

        val_accuracy = (
            100.0
            * val_correct
            / val_total
        )


        # ====================================================
        # EPOCH TIME
        # ====================================================

        epoch_time = (
            time.time()
            - epoch_start_time
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        log(
            f"Train | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy:.2f}%"
        )

        log(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.2f}%"
        )

        log(
            f"Epoch time: "
            f"{epoch_time:.2f} seconds"
        )


        # ====================================================
        # SAVE CSV AFTER EVERY EPOCH
        # ====================================================

        with open(
            csv_log_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                epoch_time
            ])


        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            best_val_loss = (
                val_loss
            )

            best_epoch = (
                epoch + 1
            )

            checkpoint = {
                "model_name":
                    MODEL_NAME,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch + 1,

                "best_val_accuracy":
                    best_val_accuracy,

                "best_val_loss":
                    best_val_loss,

                "class_to_idx":
                    full_dataset.class_to_idx,

                "num_classes":
                    NUM_CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "optimizer":
                    "AdamW",

                "random_seed":
                    RANDOM_SEED,

                "normalization_mean": [
                    0.485,
                    0.456,
                    0.406
                ],

                "normalization_std": [
                    0.229,
                    0.224,
                    0.225
                ]
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            log(
                f">>> New best model: "
                f"{best_val_accuracy:.2f}%"
            )

            log(
                f">>> Best checkpoint "
                f"saved immediately to: "
                f"{checkpoint_path}"
            )

        log("")


    # ========================================================
    # TOTAL TRAINING TIME
    # ========================================================

    total_training_time = (
        time.time()
        - training_start_time
    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_checkpoint = {
        "model_name":
            MODEL_NAME,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            EPOCHS,

        "best_epoch":
            best_epoch,

        "best_val_accuracy":
            best_val_accuracy,

        "best_val_loss":
            best_val_loss,

        "class_to_idx":
            full_dataset.class_to_idx,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "optimizer":
            "AdamW",

        "random_seed":
            RANDOM_SEED,

        "normalization_mean": [
            0.485,
            0.456,
            0.406
        ],

        "normalization_std": [
            0.229,
            0.224,
            0.225
        ]
    }

    torch.save(
        final_checkpoint,
        final_checkpoint_path
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    log("=" * 70)
    log("TRAINING COMPLETED")
    log("=" * 70)

    log(
        f"Model: {MODEL_NAME}"
    )

    log(
        "Optimizer: AdamW"
    )

    log(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    log(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    log(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    log(
        f"Best epoch: "
        f"{best_epoch}"
    )

    log(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    log(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    log(
        f"Total parameters: "
        f"{total_params:,}"
    )

    log(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    log(
        f"Training time: "
        f"{total_training_time:.2f} seconds"
    )


    # ========================================================
    # BEST CHECKPOINT SIZE
    # ========================================================

    if os.path.exists(
        checkpoint_path
    ):

        model_size_mb = (
            os.path.getsize(
                checkpoint_path
            )
            / (1024 * 1024)
        )

        log(
            f"Saved best model size: "
            f"{model_size_mb:.2f} MB"
        )


    # ========================================================
    # FINAL CHECKPOINT SIZE
    # ========================================================

    if os.path.exists(
        final_checkpoint_path
    ):

        final_model_size_mb = (
            os.path.getsize(
                final_checkpoint_path
            )
            / (1024 * 1024)
        )

        log(
            f"Saved final model size: "
            f"{final_model_size_mb:.2f} MB"
        )


    log(
        f"Best model saved to: "
        f"{checkpoint_path}"
    )

    log(
        f"Final model saved to: "
        f"{final_checkpoint_path}"
    )

    log(
        f"TXT log saved to: "
        f"{txt_log_path}"
    )

    log(
        f"CSV history saved to: "
        f"{csv_log_path}"
    )


# ============================================================
# ERROR HANDLING
# ============================================================

except Exception as e:

    log("")
    log("=" * 70)
    log("ERROR")
    log("=" * 70)
    log(str(e))

    raise


# ============================================================
# FINISH
# ============================================================

finally:

    run_finish = datetime.now()

    log("")
    log(
        f"Run finished: "
        f"{run_finish}"
    )

    log_file.close()