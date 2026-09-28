import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import (
    datasets,
    models,
    transforms
)

from torch.utils.data import (
    DataLoader,
    random_split
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset_images_v2/train"

MODEL_SAVE_PATH = "resnet18_radioml_v2.pth"

BATCH_SIZE = 32
EPOCHS = 15

NUM_CLASSES = 24

LEARNING_RATE = 0.001

RANDOM_SEED = 42


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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ResNet18 RadioML Training v2")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Reproducibility
    # --------------------------------------------------------

    set_seed(RANDOM_SEED)

    print(
        f"\nRandom seed: {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # 2. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # 3. Image preprocessing
    # --------------------------------------------------------

    data_transforms = transforms.Compose([

        # Images should already be 224x224.
        # Resize remains as a safety check.
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        # ImageNet normalization because
        # ResNet18 starts from ImageNet weights.
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

    # --------------------------------------------------------
    # 4. Load dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IMPORTANT:
    # This is the ACTUAL class mapping used by ImageFolder.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. Train / validation split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7. Initialize ResNet18
    # --------------------------------------------------------

    print(
        "\nInitializing pretrained ResNet18..."
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    model = model.to(device)

    # --------------------------------------------------------
    # 8. Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # 9. Best model tracking
    # --------------------------------------------------------

    best_val_acc = 0.0

    best_epoch = 0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    # --------------------------------------------------------
    # 10. Training
    # --------------------------------------------------------

    print(
        "\nStarting training..."
    )

    for epoch in range(EPOCHS):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        print(
            f"{'=' * 60}"
        )

        # ====================================================
        # TRAINING PHASE
        # ====================================================

        model.train()

        running_loss = 0.0
        running_corrects = 0

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

            outputs = model(inputs)

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

        # ====================================================
        # VALIDATION PHASE
        # ====================================================

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

                outputs = model(inputs)

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

        # ====================================================
        # RESULTS
        # ====================================================

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

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = (
                epoch + 1
            )

            best_model_weights = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            print(
                f">>> New best model: "
                f"{best_val_acc * 100:.2f}%"
            )

    # --------------------------------------------------------
    # 11. Restore best weights
    # --------------------------------------------------------

    model.load_state_dict(
        best_model_weights
    )

    # --------------------------------------------------------
    # 12. Save checkpoint
    #
    # Save not only weights but also metadata.
    # --------------------------------------------------------

    checkpoint = {

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

        "best_epoch":
            best_epoch,

        "random_seed":
            RANDOM_SEED,

        "image_size":
            (224, 224),

        "nfft":
            64,

        "noverlap":
            32,

        "spectrogram_fs":
            1000
    }

    torch.save(
        checkpoint,
        MODEL_SAVE_PATH
    )

    # --------------------------------------------------------
    # 13. Final report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )

    print(
        f"Model saved to: "
        f"Model saved to: "
        f"{MODEL_SAVE_PATH}"
    )


if __name__ == "__main__":
    main()