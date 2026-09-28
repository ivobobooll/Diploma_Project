import csv
import gc
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

HDF5_PATH = (
    "/home/lily/Downloads/DIPLOMA/dataset/"
    "GOLD_XYZ_OSC.0001_1024.hdf5"
)

RESULTS_DIR = Path("independent_test_results_240")

TABLES_DIR = RESULTS_DIR / "tables"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
PLOTS_DIR = RESULTS_DIR / "plots"

NUM_CLASSES = 24

MIN_TRAIN_SNR = 10
TRAIN_SAMPLES_PER_CLASS = 5000

# ------------------------------------------------------------
# MAIN CHANGE:
# 10 independent samples for every one of the 24 classes.
# ------------------------------------------------------------

TEST_SAMPLES_PER_CLASS = 10

# Total:
# 24 classes × 10 = 240 independent test samples.

RANDOM_SEED = 42

IMAGE_SIZE = 224

NFFT = 64
NOVERLAP = 32
SPECTROGRAM_FS = 1000


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "ResNet18":
        "resnet18_radioml_v2.pth",

    "MobileNetV3-Large":
        "trained_models/"
        "mobilenet_v3_large_radioml_best.pth",

    "DenseNet121":
        "trained_models/"
        "densenet121_radioml_best.pth",

    "ConvNeXt-Tiny":
        "trained_models/"
        "convnext_tiny_radioml_best.pth",

    "ViT-B/16":
        "trained_models/"
        "vit_b_16_radioml_best.pth",
}


# ============================================================
# ORIGINAL RADIOML CLASS ORDER
# ============================================================

CLASSES = [
    "OOK",
    "ASK4",
    "ASK8",
    "BPSK",
    "QPSK",
    "PSK8",
    "PSK16",
    "PSK32",
    "APSK16",
    "APSK32",
    "APSK64",
    "APSK128",
    "QAM16",
    "QAM32",
    "QAM64",
    "QAM128",
    "QAM256",
    "AM_SSB_WC",
    "AM_SSB_SC",
    "AM_DSB_WC",
    "AM_DSB_SC",
    "FM",
    "GMSK",
    "OQPSK",
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],

        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    for directory in [
        RESULTS_DIR,
        TABLES_DIR,
        CONFUSION_DIR,
        PLOTS_DIR,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print("\nChecking required files...")

    if not Path(HDF5_PATH).exists():

        raise FileNotFoundError(
            f"HDF5 not found:\n{HDF5_PATH}"
        )

    for model_name, model_path in MODEL_PATHS.items():

        if not Path(model_path).exists():

            raise FileNotFoundError(
                f"{model_name} checkpoint not found:\n"
                f"{model_path}"
            )

        print(
            f"[OK] {model_name}: {model_path}"
        )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(model_name):

    if model_name == "ResNet18":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "MobileNetV3-Large":

        model = models.mobilenet_v3_large(
            weights=None
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            NUM_CLASSES
        )

    elif model_name == "DenseNet121":

        model = models.densenet121(
            weights=None
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES
        )

    elif model_name == "ConvNeXt-Tiny":

        model = models.convnext_tiny(
            weights=None
        )

        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            NUM_CLASSES
        )

    elif model_name == "ViT-B/16":

        model = models.vit_b_16(
            weights=None
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# LOAD ONE MODEL
# ============================================================

def load_model(
    model_name,
    model_path,
    device
):

    print(
        f"\nLoading {model_name}..."
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        class_to_idx = checkpoint.get(
            "class_to_idx"
        )

        checkpoint_classes = checkpoint.get(
            "classes"
        )

    else:

        state_dict = checkpoint
        class_to_idx = None
        checkpoint_classes = None

    model = create_model(
        model_name
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Restore exact output class mapping.
    # --------------------------------------------------------

    if class_to_idx is not None:

        output_classes = [
            None
        ] * NUM_CLASSES

        for class_name, index in class_to_idx.items():

            output_classes[
                int(index)
            ] = class_name

        if any(
            class_name is None
            for class_name in output_classes
        ):

            raise ValueError(
                f"Incomplete class_to_idx "
                f"in {model_name} checkpoint."
            )

    elif checkpoint_classes is not None:

        output_classes = list(
            checkpoint_classes
        )

    else:

        # ImageFolder uses alphabetical order.
        output_classes = sorted(
            CLASSES
        )

    if len(output_classes) != NUM_CLASSES:

        raise ValueError(
            f"{model_name}: expected "
            f"{NUM_CLASSES} classes, "
            f"got {len(output_classes)}."
        )

    print(
        f"[OK] {model_name} loaded."
    )

    return (
        model,
        output_classes
    )


# ============================================================
# RECONSTRUCT EXACT ORIGINAL TRAINING INDICES
# ============================================================

def reconstruct_training_indices(
    y_dataset,
    z_dataset
):

    print(
        "\nReconstructing original "
        "training/validation source indices..."
    )

    counters = np.zeros(
        NUM_CLASSES,
        dtype=np.int32
    )

    used_indices = set()

    total_samples = len(
        y_dataset
    )

    for index in range(
        total_samples
    ):

        snr = int(
            np.asarray(
                z_dataset[index]
            ).reshape(-1)[0]
        )

        if snr < MIN_TRAIN_SNR:
            continue

        class_index = int(
            np.argmax(
                y_dataset[index]
            )
        )

        if (
            counters[class_index]
            >= TRAIN_SAMPLES_PER_CLASS
        ):
            continue

        used_indices.add(
            index
        )

        counters[
            class_index
        ] += 1

        if np.all(
            counters
            >= TRAIN_SAMPLES_PER_CLASS
        ):
            break

    print(
        "\nReconstructed samples:"
    )

    for class_index, class_name in enumerate(
        CLASSES
    ):

        print(
            f"{class_name:<12}: "
            f"{counters[class_index]}"
        )

    print(
        f"\nTotal reconstructed: "
        f"{len(used_indices)}"
    )

    expected_total = (
        NUM_CLASSES
        *
        TRAIN_SAMPLES_PER_CLASS
    )

    if len(used_indices) != expected_total:

        raise RuntimeError(
            "Could not reconstruct exact "
            "120,000 source indices."
        )

    if not np.all(
        counters
        == TRAIN_SAMPLES_PER_CLASS
    ):

        raise RuntimeError(
            "At least one class does not "
            "contain exactly 5000 "
            "reconstructed samples."
        )

    print(
        "\n[OK] Exact 120,000 source "
        "indices reconstructed."
    )

    return used_indices


# ============================================================
# SELECT 240 INDEPENDENT TEST SAMPLES
# ============================================================

def select_independent_test(
    y_dataset,
    z_dataset,
    used_indices
):

    print(
        "\nSelecting 240 independent "
        "test samples..."
    )

    # Separate candidate indices by class.
    candidates = [
        []
        for _ in range(
            NUM_CLASSES
        )
    ]

    total_samples = len(
        y_dataset
    )

    for index in range(
        total_samples
    ):

        # ----------------------------------------------------
        # CRITICAL:
        # Never use a source I/Q record that was used
        # for training OR validation.
        # ----------------------------------------------------

        if index in used_indices:
            continue

        snr = int(
            np.asarray(
                z_dataset[index]
            ).reshape(-1)[0]
        )

        # Main independent test uses the same SNR condition
        # as the training dataset.
        if snr < MIN_TRAIN_SNR:
            continue

        class_index = int(
            np.argmax(
                y_dataset[index]
            )
        )

        candidates[
            class_index
        ].append(
            index
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    test_samples = []

    for class_index in range(
        NUM_CLASSES
    ):

        class_candidates = np.asarray(
            candidates[
                class_index
            ],
            dtype=np.int64
        )

        if (
            len(class_candidates)
            < TEST_SAMPLES_PER_CLASS
        ):

            raise RuntimeError(
                f"Not enough independent samples "
                f"for class "
                f"{CLASSES[class_index]}."
            )

        selected = rng.choice(
            class_candidates,
            size=TEST_SAMPLES_PER_CLASS,
            replace=False
        )

        for index in selected:

            index = int(
                index
            )

            snr = int(
                np.asarray(
                    z_dataset[index]
                ).reshape(-1)[0]
            )

            test_samples.append({
                "hdf5_index":
                    index,

                "class_index":
                    class_index,

                "class_name":
                    CLASSES[
                        class_index
                    ],

                "snr":
                    snr,
            })

    # --------------------------------------------------------
    # Shuffle complete balanced test set.
    # --------------------------------------------------------

    permutation = rng.permutation(
        len(test_samples)
    )

    test_samples = [
        test_samples[index]
        for index in permutation
    ]

    # --------------------------------------------------------
    # STRICT OVERLAP CHECK
    # --------------------------------------------------------

    test_indices = {
        sample["hdf5_index"]
        for sample in test_samples
    }

    overlap = (
        test_indices
        &
        used_indices
    )

    print(
        f"[OK] Independent test size: "
        f"{len(test_samples)}"
    )

    print(
        "[OK] Overlap with original "
        "training/validation source "
        f"samples: {len(overlap)}"
    )

    if len(test_samples) != 240:

        raise RuntimeError(
            "Independent test does not "
            "contain exactly 240 samples."
        )

    if len(overlap) != 0:

        raise RuntimeError(
            "TEST/TRAIN OVERLAP DETECTED. "
            "Evaluation aborted."
        )

    # --------------------------------------------------------
    # Verify 10 samples per class.
    # --------------------------------------------------------

    class_counts = defaultdict(
        int
    )

    for sample in test_samples:

        class_counts[
            sample["class_name"]
        ] += 1

    for class_name in CLASSES:

        if (
            class_counts[class_name]
            != TEST_SAMPLES_PER_CLASS
        ):

            raise RuntimeError(
                f"Class {class_name} does not "
                f"contain exactly "
                f"{TEST_SAMPLES_PER_CLASS} "
                f"test samples."
            )

    print(
        "[OK] Exactly 10 independent "
        "samples selected for each class."
    )

    return test_samples


# ============================================================
# SAVE TEST METADATA
# ============================================================

def save_metadata(
    test_samples
):

    path = (
        RESULTS_DIR
        /
        "independent_test_metadata.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "test_number",
            "hdf5_index",
            "class",
            "snr_db",
        ])

        for number, sample in enumerate(
            test_samples,
            start=1
        ):

            writer.writerow([
                number,
                sample["hdf5_index"],
                sample["class_name"],
                sample["snr"],
            ])

    print(
        f"[OK] Metadata saved: {path}"
    )


# ============================================================
# IQ -> EXACT 224x224 SPECTROGRAM TENSOR
# ============================================================

def iq_to_tensor(
    iq_data
):

    # --------------------------------------------------------
    # HDF5 X shape:
    # [1024, 2]
    #
    # column 0 = I
    # column 1 = Q
    # --------------------------------------------------------

    samples = (
        iq_data[:, 0]
        +
        1j
        *
        iq_data[:, 1]
    )

    fig = plt.Figure(
        figsize=(2.24, 2.24),
        dpi=100
    )

    canvas = FigureCanvasAgg(
        fig
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    ax.axis(
        "off"
    )

    ax.specgram(
        samples,
        NFFT=NFFT,
        Fs=SPECTROGRAM_FS,
        noverlap=NOVERLAP,
        cmap="viridis"
    )

    canvas.draw()

    rgba = np.asarray(
        canvas.buffer_rgba()
    )

    rgb = np.asarray(
        rgba[:, :, :3]
    ).copy()

    plt.close(
        fig
    )

    image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    if image.size != (
        IMAGE_SIZE,
        IMAGE_SIZE
    ):

        raise RuntimeError(
            f"Generated spectrogram size "
            f"is {image.size}, expected "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    tensor = IMAGE_TRANSFORM(
        image
    )

    return tensor


# ============================================================
# CACHE ONLY 240 TEST TENSORS
# ============================================================

def create_test_cache(
    x_dataset,
    test_samples
):

    print(
        "\nCreating 240 spectrograms..."
    )

    tensors = []

    for number, sample in enumerate(
        test_samples,
        start=1
    ):

        iq_data = np.asarray(
            x_dataset[
                sample["hdf5_index"]
            ]
        )

        tensor = iq_to_tensor(
            iq_data
        )

        # Keep tensor on CPU.
        tensors.append(
            tensor
        )

        if (
            number % 40 == 0
            or number
            == len(test_samples)
        ):

            print(
                f"  {number:>3} / "
                f"{len(test_samples)}"
            )

    # --------------------------------------------------------
    # Stack:
    # [240, 3, 224, 224]
    #
    # Roughly 145 MB float32.
    # --------------------------------------------------------

    cache = torch.stack(
        tensors,
        dim=0
    )

    del tensors

    gc.collect()

    print(
        "[OK] 240 spectrograms created "
        "once and cached in CPU memory."
    )

    return cache


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model,
    output_classes,
    test_cache,
    test_samples,
    device
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"EVALUATING: {model_name}"
    )

    print(
        "=" * 70
    )

    y_true_names = []
    y_pred_names = []

    prediction_rows = []

    batch_size = 16

    total = len(
        test_samples
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = (
            test_cache[
                start:end
            ]
            .to(device)
        )

        with torch.inference_mode():

            outputs = model(
                batch
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top3_probabilities, top3_indices = (
                torch.topk(
                    probabilities,
                    k=3,
                    dim=1
                )
            )

        top3_probabilities = (
            top3_probabilities
            .cpu()
            .numpy()
        )

        top3_indices = (
            top3_indices
            .cpu()
            .numpy()
        )

        for local_index in range(
            end - start
        ):

            sample = test_samples[
                start
                +
                local_index
            ]

            true_class = sample[
                "class_name"
            ]

            predicted_classes = [
                output_classes[
                    int(index)
                ]
                for index
                in top3_indices[
                    local_index
                ]
            ]

            predicted_probabilities = (
                top3_probabilities[
                    local_index
                ]
                *
                100
            )

            predicted_class = (
                predicted_classes[0]
            )

            y_true_names.append(
                true_class
            )

            y_pred_names.append(
                predicted_class
            )

            prediction_rows.append({
                "hdf5_index":
                    sample["hdf5_index"],

                "true_class":
                    true_class,

                "snr":
                    sample["snr"],

                "top1":
                    predicted_classes[0],

                "top1_probability":
                    float(
                        predicted_probabilities[0]
                    ),

                "top2":
                    predicted_classes[1],

                "top2_probability":
                    float(
                        predicted_probabilities[1]
                    ),

                "top3":
                    predicted_classes[2],

                "top3_probability":
                    float(
                        predicted_probabilities[2]
                    ),
            })

        del batch
        del outputs
        del probabilities

        if device.type == "cuda":

            torch.cuda.empty_cache()

        print(
            f"  {end:>3} / {total}"
        )

    return (
        y_true_names,
        y_pred_names,
        prediction_rows
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0
    )

    return {
        "Accuracy":
            accuracy,

        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1,

        "Weighted Precision":
            weighted_precision,

        "Weighted Recall":
            weighted_recall,

        "Weighted F1":
            weighted_f1,
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    model_name,
    rows
):

    safe_name = (
        model_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    path = (
        TABLES_DIR
        /
        f"{safe_name}_predictions.csv"
    )

    fieldnames = [
        "hdf5_index",
        "true_class",
        "snr",
        "top1",
        "top1_probability",
        "top2",
        "top2_probability",
        "top3",
        "top3_probability",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"[OK] Predictions saved: {path}"
    )


# ============================================================
# PER-CLASS METRICS
# ============================================================

def calculate_per_class_metrics(
    model_name,
    y_true,
    y_pred
):

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average=None,
        zero_division=0
    )

    rows = []

    for index, class_name in enumerate(
        CLASSES
    ):

        class_true = np.array(
            y_true
        ) == class_name

        class_pred = np.array(
            y_pred
        )

        correct = int(
            np.sum(
                class_pred[
                    class_true
                ]
                ==
                class_name
            )
        )

        total = int(
            np.sum(
                class_true
            )
        )

        class_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        rows.append({
            "Model":
                model_name,

            "Class":
                class_name,

            "Correct":
                correct,

            "Total":
                total,

            "Class Accuracy":
                class_accuracy,

            "Precision":
                precision[index],

            "Recall":
                recall[index],

            "F1":
                f1[index],

            "Support":
                int(
                    support[index]
                ),
        })

    return rows


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    model_name,
    y_true,
    y_pred
):

    safe_name = (
        model_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    # --------------------------------------------------------
    # RAW CSV
    # --------------------------------------------------------

    csv_path = (
        CONFUSION_DIR
        /
        f"{safe_name}_confusion_matrix_raw.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            ["True / Predicted"]
            +
            CLASSES
        )

        for class_name, row in zip(
            CLASSES,
            matrix
        ):

            writer.writerow(
                [class_name]
                +
                row.tolist()
            )

    # --------------------------------------------------------
    # NORMALIZED MATRIX
    # --------------------------------------------------------

    row_sums = matrix.sum(
        axis=1,
        keepdims=True
    )

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=float
        ),
        where=row_sums != 0
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    image = ax.imshow(
        normalized,
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_title(
        f"Normalized Confusion Matrix — "
        f"{model_name}",
        fontsize=15
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_xticks(
        np.arange(
            NUM_CLASSES
        )
    )

    ax.set_yticks(
        np.arange(
            NUM_CLASSES
        )
    )

    ax.set_xticklabels(
        CLASSES,
        rotation=90,
        fontsize=7
    )

    ax.set_yticklabels(
        CLASSES,
        fontsize=7
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Fraction"
    )

    fig.tight_layout()

    png_path = (
        CONFUSION_DIR
        /
        f"{safe_name}_confusion_matrix.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] Confusion matrix saved."
    )


# ============================================================
# SNR ROBUSTNESS FROM THE SAME 240 TEST SAMPLES
# ============================================================

def calculate_snr_results(
    model_name,
    prediction_rows
):

    grouped = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    for row in prediction_rows:

        snr = int(
            row["snr"]
        )

        grouped[
            snr
        ][
            "total"
        ] += 1

        if (
            row["true_class"]
            ==
            row["top1"]
        ):

            grouped[
                snr
            ][
                "correct"
            ] += 1

    rows = []

    for snr in sorted(
        grouped.keys()
    ):

        correct = grouped[
            snr
        ][
            "correct"
        ]

        total = grouped[
            snr
        ][
            "total"
        ]

        accuracy = (
            correct
            /
            total
        )

        rows.append({
            "Model":
                model_name,

            "SNR":
                snr,

            "Correct":
                correct,

            "Total":
                total,

            "Accuracy":
                accuracy,
        })

    return rows


# ============================================================
# MODEL SIZE
# ============================================================

def calculate_state_dict_size_mb(
    model
):

    with tempfile.NamedTemporaryFile(
        suffix=".pth"
    ) as temporary_file:

        torch.save(
            model.state_dict(),
            temporary_file.name
        )

        temporary_file.flush()

        size_bytes = Path(
            temporary_file.name
        ).stat().st_size

    return (
        size_bytes
        /
        (1024 ** 2)
    )


# ============================================================
# INFERENCE TIME
# ============================================================

def measure_inference_time(
    model,
    device
):

    dummy = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device
    )

    # Warm-up.
    with torch.inference_mode():

        for _ in range(10):

            _ = model(
                dummy
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    repeats = 50

    start = time.perf_counter()

    with torch.inference_mode():

        for _ in range(
            repeats
        ):

            _ = model(
                dummy
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    end = time.perf_counter()

    average_ms = (
        (end - start)
        /
        repeats
        *
        1000
    )

    del dummy

    return average_ms


# ============================================================
# SAVE CSV
# ============================================================

def save_dict_rows(
    path,
    rows
):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# PRETTY METRICS TABLE
# ============================================================

def create_metrics_table_png(
    metrics_rows
):

    columns = [
        "Model",
        "Accuracy",
        "Macro\nPrecision",
        "Macro\nRecall",
        "Macro\nF1",
        "Weighted\nF1",
    ]

    data = []

    for row in metrics_rows:

        data.append([
            row["Model"],

            f"{row['Accuracy'] * 100:.2f}%",

            f"{row['Macro Precision'] * 100:.2f}%",

            f"{row['Macro Recall'] * 100:.2f}%",

            f"{row['Macro F1'] * 100:.2f}%",

            f"{row['Weighted F1'] * 100:.2f}%",
        ])

    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    ax.axis(
        "off"
    )

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        10
    )

    table.scale(
        1,
        1.8
    )

    for column in range(
        len(columns)
    ):

        table[
            0,
            column
        ].set_text_props(
            weight="bold"
        )

    ax.set_title(
        "Independent RadioML Test — "
        "240 Unseen Samples",
        fontsize=15,
        pad=20
    )

    fig.tight_layout()

    fig.savefig(
        TABLES_DIR
        /
        "model_metrics_table.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# COMPUTATIONAL TABLE
# ============================================================

def create_computational_table_png(
    rows
):

    columns = [
        "Model",
        "Parameters",
        "State dict,\nMB",
        "Inference,\nms/sample",
    ]

    data = []

    for row in rows:

        data.append([
            row["Model"],

            f"{row['Parameters']:,}",

            f"{row['State Dict Size MB']:.2f}",

            f"{row['Inference Time ms']:.3f}",
        ])

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    ax.axis(
        "off"
    )

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        10
    )

    table.scale(
        1,
        1.8
    )

    for column in range(
        len(columns)
    ):

        table[
            0,
            column
        ].set_text_props(
            weight="bold"
        )

    ax.set_title(
        "Computational Characteristics",
        fontsize=15,
        pad=20
    )

    fig.tight_layout()

    fig.savefig(
        TABLES_DIR
        /
        "computational_metrics_table.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# COMPARISON BAR PLOT
# ============================================================

def create_metric_bar_plot(
    metrics_rows,
    metric,
    filename,
    title
):

    model_names = [
        row["Model"]
        for row in metrics_rows
    ]

    values = [
        row[metric]
        *
        100
        for row in metrics_rows
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        model_names,
        values
    )

    ax.set_ylabel(
        "Percent, %"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        title
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            value + 1,

            f"{value:.1f}%",

            ha="center"
        )

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR
        /
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# SNR PLOT
# ============================================================

def create_snr_plot(
    all_snr_rows
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model_name in MODEL_PATHS.keys():

        model_rows = [
            row
            for row in all_snr_rows
            if row["Model"]
            == model_name
        ]

        model_rows = sorted(
            model_rows,
            key=lambda row:
                row["SNR"]
        )

        x = [
            row["SNR"]
            for row in model_rows
        ]

        y = [
            row["Accuracy"]
            *
            100
            for row in model_rows
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=model_name
        )

    ax.set_xlabel(
        "SNR, dB"
    )

    ax.set_ylabel(
        "Accuracy, %"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "Classification Accuracy vs SNR "
        "— Independent Test"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR
        /
        "accuracy_vs_snr_all_models.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "INDEPENDENT RADIOML MODEL "
        "EVALUATION - 240 SAMPLES"
    )

    print(
        "=" * 70
    )

    create_directories()

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU: "
            +
            torch.cuda.get_device_name(
                0
            )
        )

    check_required_files()

    # --------------------------------------------------------
    # OPEN HDF5
    # --------------------------------------------------------

    print(
        "\nOpening RadioML HDF5..."
    )

    with h5py.File(
        HDF5_PATH,
        "r"
    ) as h5_file:

        print(
            f"HDF5 keys: "
            f"{list(h5_file.keys())}"
        )

        x_dataset = h5_file[
            "X"
        ]

        y_dataset = h5_file[
            "Y"
        ]

        z_dataset = h5_file[
            "Z"
        ]

        print(
            f"X shape: "
            f"{x_dataset.shape}"
        )

        print(
            f"Y shape: "
            f"{y_dataset.shape}"
        )

        print(
            f"Z shape: "
            f"{z_dataset.shape}"
        )

        # ----------------------------------------------------
        # EXACT TRAIN/VAL SOURCE RECONSTRUCTION
        # ----------------------------------------------------

        used_indices = (
            reconstruct_training_indices(
                y_dataset,
                z_dataset
            )
        )

        # ----------------------------------------------------
        # SELECT ONLY 240 UNSEEN SIGNALS
        # ----------------------------------------------------

        test_samples = (
            select_independent_test(
                y_dataset,
                z_dataset,
                used_indices
            )
        )

        save_metadata(
            test_samples
        )

        # ----------------------------------------------------
        # GENERATE EACH SPECTROGRAM ONLY ONCE
        # ----------------------------------------------------

        test_cache = (
            create_test_cache(
                x_dataset,
                test_samples
            )
        )

    # HDF5 can now be closed.
    del used_indices

    gc.collect()

    # --------------------------------------------------------
    # EVALUATE MODELS ONE AT A TIME
    # --------------------------------------------------------

    all_metrics = []
    all_per_class = []
    all_snr_rows = []
    computational_rows = []

    for (
        model_name,
        model_path
    ) in MODEL_PATHS.items():

        model, output_classes = (
            load_model(
                model_name,
                model_path,
                device
            )
        )

        (
            y_true,
            y_pred,
            prediction_rows
        ) = evaluate_model(
            model_name,
            model,
            output_classes,
            test_cache,
            test_samples,
            device
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        metrics_row = {
            "Model":
                model_name,

            **metrics,
        }

        all_metrics.append(
            metrics_row
        )

        print(
            f"\n{model_name} results:"
        )

        print(
            f"  Accuracy: "
            f"{metrics['Accuracy'] * 100:.2f}%"
        )

        print(
            f"  Macro Precision: "
            f"{metrics['Macro Precision'] * 100:.2f}%"
        )

        print(
            f"  Macro Recall: "
            f"{metrics['Macro Recall'] * 100:.2f}%"
        )

        print(
            f"  Macro F1: "
            f"{metrics['Macro F1'] * 100:.2f}%"
        )

        print(
            f"  Weighted F1: "
            f"{metrics['Weighted F1'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS IMMEDIATELY
        # ----------------------------------------------------

        save_predictions(
            model_name,
            prediction_rows
        )

        # ----------------------------------------------------
        # PER CLASS
        # ----------------------------------------------------

        per_class_rows = (
            calculate_per_class_metrics(
                model_name,
                y_true,
                y_pred
            )
        )

        all_per_class.extend(
            per_class_rows
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        save_confusion_matrix(
            model_name,
            y_true,
            y_pred
        )

        # ----------------------------------------------------
        # SNR ROBUSTNESS
        #
        # IMPORTANT:
        # Uses the SAME 240 independent samples.
        # No extra 30,000 samples are generated.
        # ----------------------------------------------------

        snr_rows = (
            calculate_snr_results(
                model_name,
                prediction_rows
            )
        )

        all_snr_rows.extend(
            snr_rows
        )

        # ----------------------------------------------------
        # COMPUTATIONAL CHARACTERISTICS
        # ----------------------------------------------------

        parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )

        state_dict_size = (
            calculate_state_dict_size_mb(
                model
            )
        )

        inference_time = (
            measure_inference_time(
                model,
                device
            )
        )

        computational_rows.append({
            "Model":
                model_name,

            "Parameters":
                parameters,

            "State Dict Size MB":
                state_dict_size,

            "Inference Time ms":
                inference_time,
        })

        # ----------------------------------------------------
        # SAVE CURRENT CUMULATIVE RESULTS IMMEDIATELY
        # ----------------------------------------------------

        save_dict_rows(
            TABLES_DIR
            /
            "model_metrics.csv",
            all_metrics
        )

        save_dict_rows(
            TABLES_DIR
            /
            "per_class_metrics.csv",
            all_per_class
        )

        save_dict_rows(
            TABLES_DIR
            /
            "snr_accuracy.csv",
            all_snr_rows
        )

        save_dict_rows(
            TABLES_DIR
            /
            "computational_metrics.csv",
            computational_rows
        )

        # ----------------------------------------------------
        # REMOVE MODEL BEFORE NEXT ONE
        # ----------------------------------------------------

        del model

        gc.collect()

        if device.type == "cuda":

            torch.cuda.empty_cache()

        print(
            f"\n[OK] {model_name} finished "
            "and removed from GPU memory."
        )

    # --------------------------------------------------------
    # FINAL TABLES
    # --------------------------------------------------------

    create_metrics_table_png(
        all_metrics
    )

    create_computational_table_png(
        computational_rows
    )

    # --------------------------------------------------------
    # COMPARISON PLOTS
    # --------------------------------------------------------

    create_metric_bar_plot(
        all_metrics,
        "Accuracy",
        "accuracy_comparison.png",
        (
            "Independent Test Accuracy "
            "— 240 Unseen RadioML Samples"
        )
    )

    create_metric_bar_plot(
        all_metrics,
        "Macro F1",
        "macro_f1_comparison.png",
        (
            "Independent Test Macro F1 "
            "— 240 Unseen RadioML Samples"
        )
    )

    # --------------------------------------------------------
    # SNR PLOT
    # --------------------------------------------------------

    create_snr_plot(
        all_snr_rows
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    for row in all_metrics:

        print(
            f"\n{row['Model']}"
        )

        print(
            f"  Accuracy: "
            f"{row['Accuracy'] * 100:.2f}%"
        )

        print(
            f"  Macro F1: "
            f"{row['Macro F1'] * 100:.2f}%"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults directory:\n"
        f"  {RESULTS_DIR.resolve()}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Independent test samples: 240"
    )

    print(
        "  Classes: 24"
    )

    print(
        "  Samples per class: 10"
    )

    print(
        "  Train/validation overlap: 0"
    )

    print(
        "  Same 240 samples used by all models."
    )

    print(
        "  SNR analysis uses the same 240 samples."
    )

    print(
        "\nDone."
    )


# ============================================================
# RUN
# ============================================================



# ============================================================
# FINAL MULTI-TEST EXTENSION
# Test 1 is preserved from the already completed run.
# Tests 2 and 3 use new, mutually disjoint balanced samples.
# Test 4 uses 500 new FM-only samples.
# ============================================================

ORIGINAL_TEST1_DIR = Path("independent_test_results_240")
FINAL_ROOT = Path("independent_tests_final")
TEST2_DIR = FINAL_ROOT / "test_2"
TEST3_DIR = FINAL_ROOT / "test_3"
FM_DIR = FINAL_ROOT / "test_4_fm_500"
FINAL_COMPARISON_DIR = FINAL_ROOT / "final_comparison"

TEST2_SEED = 123
TEST3_SEED = 2026
FM_SEED = 777
FM_TEST_SAMPLES = 500
FM_CLASS_NAME = "FM"


def configure_output_dirs(base_dir):
    """Redirect the original save helpers to a dedicated test folder."""
    global RESULTS_DIR, TABLES_DIR, CONFUSION_DIR, PLOTS_DIR
    RESULTS_DIR = Path(base_dir)
    TABLES_DIR = RESULTS_DIR / "tables"
    CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
    PLOTS_DIR = RESULTS_DIR / "plots"
    create_directories()


def read_metadata_indices(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required metadata not found: {path}\n"
            "Keep the completed Test 1 folder independent_test_results_240."
        )
    indices = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.add(int(row["hdf5_index"]))
    return indices


def select_balanced_test(y_dataset, z_dataset, forbidden_indices, seed, samples_per_class=10):
    """Select a balanced unseen test set, excluding every forbidden HDF5 index."""
    candidates = [[] for _ in range(NUM_CLASSES)]
    total_samples = len(y_dataset)

    print(f"\nSelecting balanced test: {samples_per_class} samples/class, seed={seed}...")
    for index in range(total_samples):
        if index in forbidden_indices:
            continue
        snr = int(np.asarray(z_dataset[index]).reshape(-1)[0])
        if snr < MIN_TRAIN_SNR:
            continue
        class_index = int(np.argmax(y_dataset[index]))
        candidates[class_index].append(index)

    rng = np.random.default_rng(seed)
    samples = []
    for class_index in range(NUM_CLASSES):
        pool = np.asarray(candidates[class_index], dtype=np.int64)
        if len(pool) < samples_per_class:
            raise RuntimeError(f"Not enough unseen samples for {CLASSES[class_index]}.")
        chosen = rng.choice(pool, size=samples_per_class, replace=False)
        for idx in chosen:
            idx = int(idx)
            snr = int(np.asarray(z_dataset[idx]).reshape(-1)[0])
            samples.append({
                "hdf5_index": idx,
                "class_index": class_index,
                "class_name": CLASSES[class_index],
                "snr": snr,
            })

    perm = rng.permutation(len(samples))
    samples = [samples[i] for i in perm]
    selected_indices = {s["hdf5_index"] for s in samples}
    overlap = selected_indices & forbidden_indices
    if overlap:
        raise RuntimeError(f"Forbidden overlap detected: {len(overlap)} samples.")
    expected = NUM_CLASSES * samples_per_class
    if len(samples) != expected or len(selected_indices) != expected:
        raise RuntimeError("Balanced test size/uniqueness check failed.")

    counts = defaultdict(int)
    for s in samples:
        counts[s["class_name"]] += 1
    for name in CLASSES:
        if counts[name] != samples_per_class:
            raise RuntimeError(f"Class count check failed for {name}: {counts[name]}")

    print(f"[OK] Selected {len(samples)} unique independent samples.")
    print("[OK] Overlap with all forbidden samples: 0")
    return samples


def select_fm_test(y_dataset, z_dataset, forbidden_indices, seed, count=500):
    """Select new FM-only RadioML samples not used anywhere else."""
    fm_index = CLASSES.index(FM_CLASS_NAME)
    candidates = []
    print(f"\nSelecting {count} new FM-only samples, seed={seed}...")

    for index in range(len(y_dataset)):
        if index in forbidden_indices:
            continue
        snr = int(np.asarray(z_dataset[index]).reshape(-1)[0])
        if snr < MIN_TRAIN_SNR:
            continue
        class_index = int(np.argmax(y_dataset[index]))
        if class_index == fm_index:
            candidates.append(index)

    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} eligible unseen FM samples found; need {count}.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(candidates, dtype=np.int64), size=count, replace=False)
    samples = []
    for idx in chosen:
        idx = int(idx)
        snr = int(np.asarray(z_dataset[idx]).reshape(-1)[0])
        samples.append({
            "hdf5_index": idx,
            "class_index": fm_index,
            "class_name": FM_CLASS_NAME,
            "snr": snr,
        })

    selected = {s["hdf5_index"] for s in samples}
    if len(selected) != count:
        raise RuntimeError("FM uniqueness check failed.")
    if selected & forbidden_indices:
        raise RuntimeError("FM overlap with previous data detected.")

    print(f"[OK] Selected {count} unique FM samples.")
    print("[OK] FM overlap with train/val and Tests 1-3: 0")
    return samples


def save_metadata_to(samples, path, test_label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "test_number", "hdf5_index", "class", "snr_db"])
        for n, sample in enumerate(samples, start=1):
            writer.writerow([test_label, n, sample["hdf5_index"], sample["class_name"], sample["snr"]])
    print(f"[OK] Metadata saved: {path}")


def run_balanced_evaluation(test_name, base_dir, test_samples, test_cache, device):
    """Run all five models on one balanced test and save the original metrics."""
    configure_output_dirs(base_dir)
    all_metrics, all_per_class, all_snr_rows = [], [], []

    for model_name, model_path in MODEL_PATHS.items():
        model, output_classes = load_model(model_name, model_path, device)
        y_true, y_pred, prediction_rows = evaluate_model(
            model_name, model, output_classes, test_cache, test_samples, device
        )
        metrics = calculate_metrics(y_true, y_pred)
        row = {"Model": model_name, **metrics}
        all_metrics.append(row)
        print(f"\n{test_name} — {model_name}: Accuracy={metrics['Accuracy']*100:.2f}%, Macro F1={metrics['Macro F1']*100:.2f}%")

        save_predictions(model_name, prediction_rows)
        all_per_class.extend(calculate_per_class_metrics(model_name, y_true, y_pred))
        save_confusion_matrix(model_name, y_true, y_pred)
        all_snr_rows.extend(calculate_snr_results(model_name, prediction_rows))

        save_dict_rows(TABLES_DIR / "model_metrics.csv", all_metrics)
        save_dict_rows(TABLES_DIR / "per_class_metrics.csv", all_per_class)
        save_dict_rows(TABLES_DIR / "snr_accuracy.csv", all_snr_rows)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    create_metrics_table_png(all_metrics)
    create_metric_bar_plot(all_metrics, "Accuracy", "accuracy_comparison.png", f"{test_name} Accuracy — 240 Unseen RadioML Samples")
    create_metric_bar_plot(all_metrics, "Macro F1", "macro_f1_comparison.png", f"{test_name} Macro F1 — 240 Unseen RadioML Samples")
    create_snr_plot(all_snr_rows)
    return all_metrics


def run_fm_evaluation(base_dir, fm_samples, fm_cache, device):
    """Evaluate FM recognition. Top-1/Top-3 are directly comparable to later real-FM recognition rates."""
    configure_output_dirs(base_dir)
    summary_rows = []

    for model_name, model_path in MODEL_PATHS.items():
        model, output_classes = load_model(model_name, model_path, device)
        y_true, y_pred, rows = evaluate_model(
            model_name, model, output_classes, fm_cache, fm_samples, device
        )

        top1_correct = sum(row["top1"] == FM_CLASS_NAME for row in rows)
        top3_correct = sum(
            FM_CLASS_NAME in (row["top1"], row["top2"], row["top3"])
            for row in rows
        )
        total = len(rows)
        top1_rate = top1_correct / total
        top3_rate = top3_correct / total

        summary = {
            "Model": model_name,
            "FM Samples": total,
            "FM Top-1 Correct": top1_correct,
            "FM Top-1 Rate": top1_rate,
            "FM Top-3 Correct": top3_correct,
            "FM Top-3 Rate": top3_rate,
        }
        summary_rows.append(summary)
        save_predictions(model_name, rows)
        save_dict_rows(TABLES_DIR / "fm_500_summary.csv", summary_rows)

        print(f"\nFM 500 — {model_name}")
        print(f"  Top-1 FM: {top1_correct}/{total} = {top1_rate*100:.2f}%")
        print(f"  Top-3 FM: {top3_correct}/{total} = {top3_rate*100:.2f}%")

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # FM comparison plot
    names = [r["Model"] for r in summary_rows]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width/2, [r["FM Top-1 Rate"]*100 for r in summary_rows], width, label="FM Top-1")
    b2 = ax.bar(x + width/2, [r["FM Top-3 Rate"]*100 for r in summary_rows], width, label="FM Top-3")
    ax.set_ylabel("Recognition rate, %")
    ax.set_ylim(0, 100)
    ax.set_title("RadioML FM-only Test — 500 Unseen FM Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+1, f"{h:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fm_top1_top3_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def read_model_metrics_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "Model": row["Model"],
                "Accuracy": float(row["Accuracy"]),
                "Macro F1": float(row["Macro F1"]),
            })
    return rows


def create_three_test_summary(test1_metrics, test2_metrics, test3_metrics):
    FINAL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    by_test = {
        "Test 1": {r["Model"]: r for r in test1_metrics},
        "Test 2": {r["Model"]: r for r in test2_metrics},
        "Test 3": {r["Model"]: r for r in test3_metrics},
    }
    rows = []
    for model_name in MODEL_PATHS.keys():
        acc = np.array([by_test[t][model_name]["Accuracy"] for t in by_test], dtype=float)
        f1 = np.array([by_test[t][model_name]["Macro F1"] for t in by_test], dtype=float)
        rows.append({
            "Model": model_name,
            "Test 1 Accuracy": acc[0],
            "Test 2 Accuracy": acc[1],
            "Test 3 Accuracy": acc[2],
            "Mean Accuracy": float(acc.mean()),
            "Accuracy SD": float(acc.std(ddof=1)),
            "Test 1 Macro F1": f1[0],
            "Test 2 Macro F1": f1[1],
            "Test 3 Macro F1": f1[2],
            "Mean Macro F1": float(f1.mean()),
            "Macro F1 SD": float(f1.std(ddof=1)),
        })

    save_dict_rows(FINAL_COMPARISON_DIR / "three_tests_comparison.csv", rows)

    # Accuracy mean ± SD plot
    names = [r["Model"] for r in rows]
    means = [r["Mean Accuracy"]*100 for r in rows]
    sds = [r["Accuracy SD"]*100 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=sds, capsize=5)
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(0, 100)
    ax.set_title("Independent RadioML Accuracy — Mean ± SD Across 3 Tests")
    ax.tick_params(axis="x", rotation=20)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, mean+2, f"{mean:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FINAL_COMPARISON_DIR / "accuracy_mean_sd_three_tests.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "="*70)
    print("THREE-TEST SUMMARY")
    print("="*70)
    for r in rows:
        print(
            f"{r['Model']:<20} "
            f"Accuracy {r['Mean Accuracy']*100:.2f}% ± {r['Accuracy SD']*100:.2f}% | "
            f"Macro F1 {r['Mean Macro F1']*100:.2f}% ± {r['Macro F1 SD']*100:.2f}%"
        )
    return rows


def final_main():
    print("="*70)
    print("FINAL RADIOML EVALUATION: TESTS 2-3 + FM-ONLY TEST 4")
    print("="*70)
    print("Test 1 is preserved and read from independent_test_results_240.")

    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print("GPU: " + torch.cuda.get_device_name(0))

    # Check original Test 1 before doing any work.
    test1_metadata_path = ORIGINAL_TEST1_DIR / "independent_test_metadata.csv"
    test1_metrics_path = ORIGINAL_TEST1_DIR / "tables" / "model_metrics.csv"
    if not test1_metadata_path.exists() or not test1_metrics_path.exists():
        raise FileNotFoundError(
            "Completed Test 1 files are missing. Expected:\n"
            f"  {test1_metadata_path}\n  {test1_metrics_path}"
        )

    check_required_files()
    test1_indices = read_metadata_indices(test1_metadata_path)
    if len(test1_indices) != 240:
        raise RuntimeError(f"Test 1 metadata contains {len(test1_indices)} unique indices, expected 240.")
    print("[OK] Existing Test 1 found: 240 unique HDF5 indices.")

    with h5py.File(HDF5_PATH, "r") as h5_file:
        x_dataset = h5_file["X"]
        y_dataset = h5_file["Y"]
        z_dataset = h5_file["Z"]
        print(f"X shape: {x_dataset.shape}")
        print(f"Y shape: {y_dataset.shape}")
        print(f"Z shape: {z_dataset.shape}")

        train_indices = reconstruct_training_indices(y_dataset, z_dataset)
        overlap_t1_train = test1_indices & train_indices
        if overlap_t1_train:
            raise RuntimeError(f"Existing Test 1 overlaps train/val by {len(overlap_t1_train)} samples.")
        print("[OK] Existing Test 1 overlap with train/val: 0")

        # Test 2
        forbidden2 = train_indices | test1_indices
        test2_samples = select_balanced_test(y_dataset, z_dataset, forbidden2, TEST2_SEED, 10)
        test2_indices = {s["hdf5_index"] for s in test2_samples}
        save_metadata_to(test2_samples, TEST2_DIR / "independent_test_metadata.csv", "Test 2")
        test2_cache = create_test_cache(x_dataset, test2_samples)

        # Test 3
        forbidden3 = forbidden2 | test2_indices
        test3_samples = select_balanced_test(y_dataset, z_dataset, forbidden3, TEST3_SEED, 10)
        test3_indices = {s["hdf5_index"] for s in test3_samples}
        save_metadata_to(test3_samples, TEST3_DIR / "independent_test_metadata.csv", "Test 3")
        test3_cache = create_test_cache(x_dataset, test3_samples)

        # FM-only Test 4: exclude train/val + all three balanced tests.
        forbidden_fm = forbidden3 | test3_indices
        fm_samples = select_fm_test(y_dataset, z_dataset, forbidden_fm, FM_SEED, FM_TEST_SAMPLES)
        fm_indices = {s["hdf5_index"] for s in fm_samples}
        save_metadata_to(fm_samples, FM_DIR / "fm_test_metadata.csv", "Test 4 FM")
        fm_cache = create_test_cache(x_dataset, fm_samples)

        # Final cross-check before closing HDF5.
        assert not (test1_indices & test2_indices)
        assert not (test1_indices & test3_indices)
        assert not (test2_indices & test3_indices)
        assert not (fm_indices & train_indices)
        assert not (fm_indices & test1_indices)
        assert not (fm_indices & test2_indices)
        assert not (fm_indices & test3_indices)
        print("\n[OK] FINAL OVERLAP CHECK PASSED: all four test sets are mutually disjoint and unseen in train/val.")

    # Evaluate after HDF5 is closed.
    test2_metrics = run_balanced_evaluation("Test 2", TEST2_DIR, test2_samples, test2_cache, device)
    del test2_cache
    gc.collect()

    test3_metrics = run_balanced_evaluation("Test 3", TEST3_DIR, test3_samples, test3_cache, device)
    del test3_cache
    gc.collect()

    fm_summary = run_fm_evaluation(FM_DIR, fm_samples, fm_cache, device)
    del fm_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test1_metrics = read_model_metrics_csv(test1_metrics_path)
    create_three_test_summary(test1_metrics, test2_metrics, test3_metrics)

    print("\n" + "="*70)
    print("ALL FINAL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Existing Test 1 preserved at: {ORIGINAL_TEST1_DIR.resolve()}")
    print(f"New results saved at:       {FINAL_ROOT.resolve()}")
    print("\nGenerated:")
    print("  Test 2: 240 samples, 10/class")
    print("  Test 3: 240 samples, 10/class")
    print("  Test 4: 500 FM-only samples")
    print("  Train/val overlap: 0")
    print("  Cross-test overlap: 0")
    print("  Final Test1/Test2/Test3 mean ± SD table created")
    print("  FM Top-1/Top-3 table created for later RTL-SDR comparison")


if __name__ == "__main__":
    final_main()