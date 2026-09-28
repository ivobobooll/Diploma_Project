import copy
import csv
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset_images_v2/train"

# Choose ONE model:
#
# "resnet18"
# "mobilenet_v3_large"
# "densenet121"
# "convnext_tiny"
# "vit_b_16"
#
MODEL_NAME = "vit_b_16"

BATCH_SIZE = 32
EPOCHS = 15

NUM_CLASSES = 24

LEARNING_RATE = 0.001

RANDOM_SEED = 42

IMAGE_SIZE = 224

# Spectrogram parameters used during dataset generation
NFFT = 64
NOVERLAP = 32
SPECTROGRAM_FS = 1000

# Output folders
MODEL_DIR = "trained_models"
LOG_DIR = "training_logs"
RESULTS_DIR = "training_results"


# ============================================================
# LOGGING
# ============================================================

class Tee:
    """
    Writes console output simultaneously to:
    1. PyCharm console
    2. text log file
    """

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

        # Save log immediately.
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name, num_classes):

    print(f"\nInitializing model: {model_name}")

    # --------------------------------------------------------
    # ResNet18
    # --------------------------------------------------------

    if model_name == "resnet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        num_features = model.fc.in_features

        model.fc = nn.Linear(
            num_features,
            num_classes
        )

    # --------------------------------------------------------
    # MobileNetV3-Large
    # --------------------------------------------------------

    elif model_name == "mobilenet_v3_large":

        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

        num_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            num_features,
            num_classes
        )

    # --------------------------------------------------------
    # DenseNet121
    # --------------------------------------------------------

    elif model_name == "densenet121":

        model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        num_features = model.classifier.in_features

        model.classifier = nn.Linear(
            num_features,
            num_classes
        )

    # --------------------------------------------------------
    # ConvNeXt-Tiny
    # --------------------------------------------------------

    elif model_name == "convnext_tiny":

        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        num_features = model.classifier[2].in_features

        model.classifier[2] = nn.Linear(
            num_features,
            num_classes
        )

    # --------------------------------------------------------
    # Vision Transformer ViT-B/16
    # --------------------------------------------------------

    elif model_name == "vit_b_16":

        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )

        num_features = model.heads.head.in_features

        model.heads.head = nn.Linear(
            num_features,
            num_classes
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# COUNT PARAMETERS
# ============================================================

def count_parameters(model):

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_parameters, trainable_parameters


# ============================================================
# SAVE CSV HEADER
# ============================================================

def create_csv_file(csv_path):

    with open(
        csv_path,
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
# APPEND EPOCH TO CSV
# ============================================================

def append_epoch_to_csv(
    csv_path,
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    epoch_time
):

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{epoch_time:.2f}"
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Create output folders
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 2. Unique run name
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_name = (
        f"{MODEL_NAME}_{timestamp}"
    )

    log_path = os.path.join(
        LOG_DIR,
        f"{run_name}_training.txt"
    )

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{run_name}_history.csv"
    )

    model_save_path = os.path.join(
        MODEL_DIR,
        f"{MODEL_NAME}_radioml_best.pth"
    )

    # --------------------------------------------------------
    # 3. Start TXT logging
    # --------------------------------------------------------

    log_file = open(
        log_path,
        "w",
        encoding="utf-8",
        buffering=1
    )

    original_stdout = sys.stdout

    sys.stdout = Tee(
        original_stdout,
        log_file
    )

    try:

        print("=" * 70)
        print("RadioML Model Training")
        print("=" * 70)

        print(
            f"\nSelected model: "
            f"{MODEL_NAME}"
        )

        print(
            f"Run started: "
            f"{datetime.now()}"
        )

        print(
            f"Dataset: "
            f"{DATA_DIR}"
        )

        print(
            f"TXT log: "
            f"{log_path}"
        )

        print(
            f"CSV history: "
            f"{csv_path}"
        )

        print(
            f"Best checkpoint: "
            f"{model_save_path}"
        )

        # ----------------------------------------------------
        # 4. Reproducibility
        # ----------------------------------------------------

        set_seed(
            RANDOM_SEED
        )

        print(
            f"\nRandom seed: "
            f"{RANDOM_SEED}"
        )

        # ----------------------------------------------------
        # 5. Device
        # ----------------------------------------------------

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: "
            f"{device}"
        )

        if torch.cuda.is_available():

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ----------------------------------------------------
        # 6. Training configuration
        # ----------------------------------------------------

        print("\nTraining configuration:")

        print(
            f"Batch size: "
            f"{BATCH_SIZE}"
        )

        print(
            f"Epochs: "
            f"{EPOCHS}"
        )

        print(
            f"Learning rate: "
            f"{LEARNING_RATE}"
        )

        print(
            f"Number of classes: "
            f"{NUM_CLASSES}"
        )

        print(
            f"Image size: "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}"
        )

        # ----------------------------------------------------
        # 7. Image preprocessing
        # ----------------------------------------------------

        data_transforms = transforms.Compose([

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

        # ----------------------------------------------------
        # 8. Load dataset
        # ----------------------------------------------------

        print(
            "\nLoading dataset..."
        )

        full_dataset = datasets.ImageFolder(
            DATA_DIR,
            transform=data_transforms
        )

        print(
            f"Total images: "
            f"{len(full_dataset)}"
        )

        print(
            f"Number of classes: "
            f"{len(full_dataset.classes)}"
        )

        print(
            "\nImageFolder class mapping:"
        )

        for class_name, class_index in (
            full_dataset.class_to_idx.items()
        ):

            print(
                f"{class_index:2d} -> "
                f"{class_name}"
            )

        if (
            len(full_dataset.classes)
            != NUM_CLASSES
        ):

            raise ValueError(
                f"Expected {NUM_CLASSES} classes, "
                f"but ImageFolder found "
                f"{len(full_dataset.classes)}."
            )

        # ----------------------------------------------------
        # 9. Train / validation split
        # ----------------------------------------------------

        train_size = int(
            0.8 * len(full_dataset)
        )

        val_size = (
            len(full_dataset)
            - train_size
        )

        split_generator = (
            torch.Generator()
            .manual_seed(RANDOM_SEED)
        )

        train_dataset, val_dataset = (
            random_split(
                full_dataset,
                [
                    train_size,
                    val_size
                ],
                generator=split_generator
            )
        )

        print(
            f"\nTraining images: "
            f"{train_size}"
        )

        print(
            f"Validation images: "
            f"{val_size}"
        )

        # ----------------------------------------------------
        # 10. DataLoaders
        # ----------------------------------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        # ----------------------------------------------------
        # 11. Initialize model
        # ----------------------------------------------------

        model = create_model(
            MODEL_NAME,
            NUM_CLASSES
        )

        model = model.to(
            device
        )

        # ----------------------------------------------------
        # 12. Model parameters
        # ----------------------------------------------------

        (
            total_parameters,
            trainable_parameters
        ) = count_parameters(model)

        print(
            f"\nTotal parameters: "
            f"{total_parameters:,}"
        )

        print(
            f"Trainable parameters: "
            f"{trainable_parameters:,}"
        )

        # ----------------------------------------------------
        # 13. Loss and optimizer
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        # ----------------------------------------------------
        # 14. Best model tracking
        # ----------------------------------------------------

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_epoch = 0

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        # ----------------------------------------------------
        # 15. Create CSV
        # ----------------------------------------------------

        create_csv_file(
            csv_path
        )

        # ----------------------------------------------------
        # 16. Training
        # ----------------------------------------------------

        print(
            "\nStarting training..."
        )

        training_start_time = (
            time.time()
        )

        for epoch in range(EPOCHS):

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            )

            print(
                "=" * 70
            )

            # ================================================
            # TRAINING PHASE
            # ================================================

            model.train()

            running_loss = 0.0
            running_corrects = 0

            epoch_start_time = (
                time.time()
            )

            for inputs, labels in train_loader:

                inputs = inputs.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                optimizer.zero_grad()

                outputs = model(
                    inputs
                )

                loss = criterion(
                    outputs,
                    labels
                )

                _, predictions = torch.max(
                    outputs,
                    dim=1
                )

                loss.backward()

                optimizer.step()

                running_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                running_corrects += (
                    predictions
                    == labels
                ).sum().item()

            train_loss = (
                running_loss
                / train_size
            )

            train_acc = (
                running_corrects
                / train_size
            )

            # ================================================
            # VALIDATION PHASE
            # ================================================

            model.eval()

            validation_loss = 0.0
            validation_corrects = 0

            with torch.no_grad():

                for inputs, labels in val_loader:

                    inputs = inputs.to(
                        device,
                        non_blocking=True
                    )

                    labels = labels.to(
                        device,
                        non_blocking=True
                    )

                    outputs = model(
                        inputs
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

                    _, predictions = torch.max(
                        outputs,
                        dim=1
                    )

                    validation_loss += (
                        loss.item()
                        * inputs.size(0)
                    )

                    validation_corrects += (
                        predictions
                        == labels
                    ).sum().item()

            val_loss = (
                validation_loss
                / val_size
            )

            val_acc = (
                validation_corrects
                / val_size
            )

            epoch_time = (
                time.time()
                - epoch_start_time
            )

            # ================================================
            # SAVE HISTORY
            # ================================================

            history[
                "train_loss"
            ].append(
                train_loss
            )

            history[
                "train_accuracy"
            ].append(
                train_acc
            )

            history[
                "val_loss"
            ].append(
                val_loss
            )

            history[
                "val_accuracy"
            ].append(
                val_acc
            )

            # ================================================
            # WRITE EPOCH TO CSV IMMEDIATELY
            # ================================================

            append_epoch_to_csv(
                csv_path=csv_path,
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time=epoch_time
            )

            # ================================================
            # RESULTS
            # ================================================

            print(
                f"Train | "
                f"Loss: {train_loss:.4f} | "
                f"Accuracy: "
                f"{train_acc * 100:.2f}%"
            )

            print(
                f"Val   | "
                f"Loss: {val_loss:.4f} | "
                f"Accuracy: "
                f"{val_acc * 100:.2f}%"
            )

            print(
                f"Epoch time: "
                f"{epoch_time:.2f} seconds"
            )

            # ================================================
            # BEST CHECKPOINT
            # ================================================

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                best_val_loss = val_loss

                best_epoch = (
                    epoch + 1
                )

                best_model_weights = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                # IMPORTANT:
                # Save the best model immediately.
                # If training crashes later, this file remains.
                checkpoint = {

                    "model_name":
                        MODEL_NAME,

                    "model_state_dict":
                        best_model_weights,

                    "class_to_idx":
                        full_dataset.class_to_idx,

                    "classes":
                        full_dataset.classes,

                    "num_classes":
                        NUM_CLASSES,

                    "best_val_accuracy":
                        best_val_acc,

                    "best_val_loss":
                        best_val_loss,

                    "best_epoch":
                        best_epoch,

                    "batch_size":
                        BATCH_SIZE,

                    "epochs":
                        EPOCHS,

                    "learning_rate":
                        LEARNING_RATE,

                    "random_seed":
                        RANDOM_SEED,

                    "image_size":
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        ),

                    "nfft":
                        NFFT,

                    "noverlap":
                        NOVERLAP,

                    "spectrogram_fs":
                        SPECTROGRAM_FS,

                    "total_parameters":
                        total_parameters,

                    "trainable_parameters":
                        trainable_parameters,

                    "history":
                        history
                }

                torch.save(
                    checkpoint,
                    model_save_path
                )

                print(
                    f">>> New best model: "
                    f"{best_val_acc * 100:.2f}%"
                )

                print(
                    f">>> Best checkpoint "
                    f"saved immediately to: "
                    f"{model_save_path}"
                )

        # ----------------------------------------------------
        # 17. Training completed
        # ----------------------------------------------------

        total_training_time = (
            time.time()
            - training_start_time
        )

        # Restore best weights
        model.load_state_dict(
            best_model_weights
        )

        # ----------------------------------------------------
        # 18. Update final checkpoint
        # ----------------------------------------------------

        final_checkpoint = {

            "model_name":
                MODEL_NAME,

            "model_state_dict":
                model.state_dict(),

            "class_to_idx":
                full_dataset.class_to_idx,

            "classes":
                full_dataset.classes,

            "num_classes":
                NUM_CLASSES,

            "best_val_accuracy":
                best_val_acc,

            "best_val_loss":
                best_val_loss,

            "best_epoch":
                best_epoch,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "random_seed":
                RANDOM_SEED,

            "image_size":
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),

            "nfft":
                NFFT,

            "noverlap":
                NOVERLAP,

            "spectrogram_fs":
                SPECTROGRAM_FS,

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters,

            "training_time_seconds":
                total_training_time,

            "history":
                history
        }

        torch.save(
            final_checkpoint,
            model_save_path
        )

        # ----------------------------------------------------
        # 19. Model file size
        # ----------------------------------------------------

        model_size_mb = (
            os.path.getsize(
                model_save_path
            )
            / (1024 * 1024)
        )

        # ----------------------------------------------------
        # 20. Final report
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TRAINING COMPLETED"
        )

        print(
            "=" * 70
        )

        print(
            f"Model: "
            f"{MODEL_NAME}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.4f}"
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_acc * 100:.2f}%"
        )

        print(
            f"Total parameters: "
            f"{total_parameters:,}"
        )

        print(
            f"Trainable parameters: "
            f"{trainable_parameters:,}"
        )

        print(
            f"Training time: "
            f"{total_training_time:.2f} seconds"
        )

        print(
            f"Saved model size: "
            f"{model_size_mb:.2f} MB"
        )

        print(
            f"Best model saved to: "
            f"{model_save_path}"
        )

        print(
            f"TXT log saved to: "
            f"{log_path}"
        )

        print(
            f"CSV history saved to: "
            f"{csv_path}"
        )

        print(
            f"Run finished: "
            f"{datetime.now()}"
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TRAINING INTERRUPTED BY ERROR"
        )

        print(
            "=" * 70
        )

        print(
            f"Error: "
            f"{repr(error)}"
        )

        print(
            "\nThe TXT log, CSV history and any previously "
            "saved best checkpoint remain on disk."
        )

        raise

    # ========================================================
    # CLOSE LOG CORRECTLY
    # ========================================================

    finally:

        sys.stdout = original_stdout

        log_file.close()


if __name__ == "__main__":
    main()