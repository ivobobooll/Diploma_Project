import csv
import gc
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from scipy.signal import resample_poly
from torchvision import models, transforms


# ============================================================
# EXPERIMENT C
# RTL-SDR FM CLASSIFICATION AFTER RESAMPLING
# ============================================================

# ------------------------------------------------------------
# Input:
# Full 65536-sample I/Q captures from Experiment B.
#
# Output:
# Completely separate directory.
# ------------------------------------------------------------

INPUT_IQ_DIR = Path(
    "fm_multiwindow_experiment/iq_samples"
)

OUTPUT_DIR = Path(
    "fm_resampling_experiment"
)

TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESAMPLED_IQ_DIR = OUTPUT_DIR / "resampled_iq"

for directory in [
    OUTPUT_DIR,
    TABLES_DIR,
    PLOTS_DIR,
    RESAMPLED_IQ_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# SIGNAL SETTINGS
# ============================================================

ORIGINAL_SAMPLE_RATE = 2.048e6

DECIMATION_FACTOR = 8

TARGET_SAMPLE_RATE = (
    ORIGINAL_SAMPLE_RATE
    /
    DECIMATION_FACTOR
)

ORIGINAL_CAPTURE_SAMPLES = 65536

MODEL_WINDOW_SIZE = 1024

# After decimation:
#
# 65536 / 8 = 8192 samples
#
# 8192 / 1024 = 8 model windows.

EXPECTED_RESAMPLED_SAMPLES = (
    ORIGINAL_CAPTURE_SAMPLES
    //
    DECIMATION_FACTOR
)

NUM_WINDOWS = (
    EXPECTED_RESAMPLED_SAMPLES
    //
    MODEL_WINDOW_SIZE
)

NUM_CLASSES = 24

FM_CLASS_NAME = "FM"


# ============================================================
# SPECTROGRAM SETTINGS
# ============================================================

IMAGE_SIZE = 224

NFFT = 64
NOVERLAP = 32

# IMPORTANT:
# Keep this equal to training preprocessing.
#
# This is the plotting scale used when the training
# spectrogram images were generated.
#
# It is NOT being treated here as the physical SDR
# sample rate.
SPECTROGRAM_FS = 1000


# ============================================================
# STATIONS
# ============================================================

STATIONS = [
    {
        "name": "Army_FM",
        "frequency_mhz": 94.6,
    },
    {
        "name": "Radio_NV",
        "frequency_mhz": 96.0,
    },
    {
        "name": "Kyiv_FM",
        "frequency_mhz": 98.0,
    },
    {
        "name": "Radio_ROKS",
        "frequency_mhz": 103.6,
    },
    {
        "name": "Radio_Jazz",
        "frequency_mhz": 104.6,
    },
    {
        "name": "Kiss_FM",
        "frequency_mhz": 106.5,
    },
]

CAPTURES_PER_STATION = 2


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
# CLASSES
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
# LOAD MODEL
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
    # Restore the exact class order from the checkpoint.
    # --------------------------------------------------------

    if class_to_idx is not None:

        output_classes = [
            None
        ] * NUM_CLASSES

        for (
            class_name,
            index
        ) in class_to_idx.items():

            output_classes[
                int(index)
            ] = class_name

        if any(
            class_name is None
            for class_name in output_classes
        ):

            raise ValueError(
                f"{model_name}: incomplete "
                f"class_to_idx."
            )

    elif checkpoint_classes is not None:

        output_classes = list(
            checkpoint_classes
        )

    else:

        # Training dataset was loaded with ImageFolder.
        output_classes = sorted(
            CLASSES
        )

    if len(output_classes) != NUM_CLASSES:

        raise ValueError(
            f"{model_name}: expected "
            f"{NUM_CLASSES} output classes, "
            f"got {len(output_classes)}."
        )

    if FM_CLASS_NAME not in output_classes:

        raise ValueError(
            f"{model_name}: FM class not found."
        )

    print(
        f"[OK] {model_name} loaded."
    )

    return model, output_classes


# ============================================================
# RESAMPLE / DECIMATE I/Q
# ============================================================

def resample_iq(samples):

    samples = np.asarray(
        samples,
        dtype=np.complex64
    )

    if len(samples) != ORIGINAL_CAPTURE_SAMPLES:

        raise ValueError(
            f"Expected "
            f"{ORIGINAL_CAPTURE_SAMPLES} samples, "
            f"got {len(samples)}."
        )

    # --------------------------------------------------------
    # scipy.signal.resample_poly performs polyphase
    # resampling and applies a low-pass anti-aliasing filter.
    #
    # up=1, down=8:
    #
    # 2.048 MS/s -> 256 kS/s
    # --------------------------------------------------------

    resampled = resample_poly(
        samples,
        up=1,
        down=DECIMATION_FACTOR
    )

    resampled = np.asarray(
        resampled,
        dtype=np.complex64
    )

    if len(resampled) != EXPECTED_RESAMPLED_SAMPLES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_RESAMPLED_SAMPLES} "
            f"resampled samples, "
            f"got {len(resampled)}."
        )

    return resampled


# ============================================================
# IQ WINDOW -> SPECTROGRAM
# ============================================================

def iq_to_tensor(samples):

    samples = np.asarray(
        samples
    )

    if len(samples) != MODEL_WINDOW_SIZE:

        raise ValueError(
            f"Expected "
            f"{MODEL_WINDOW_SIZE} samples, "
            f"got {len(samples)}."
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
        rgb
    ).convert(
        "RGB"
    )

    if image.size != (
        IMAGE_SIZE,
        IMAGE_SIZE
    ):

        raise RuntimeError(
            f"Spectrogram size is {image.size}, "
            f"expected "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    tensor = IMAGE_TRANSFORM(
        image
    )

    return tensor


# ============================================================
# CREATE 8 MODEL WINDOWS
# ============================================================

def create_window_tensors(
    resampled_samples
):

    tensors = []

    print(
        f"Creating {NUM_WINDOWS} "
        f"resampled spectrogram windows..."
    )

    for window_index in range(
        NUM_WINDOWS
    ):

        start = (
            window_index
            *
            MODEL_WINDOW_SIZE
        )

        end = (
            start
            +
            MODEL_WINDOW_SIZE
        )

        window = resampled_samples[
            start:end
        ]

        tensor = iq_to_tensor(
            window
        )

        tensors.append(
            tensor
        )

        print(
            f"  Window "
            f"{window_index + 1}/"
            f"{NUM_WINDOWS}"
        )

    tensors = torch.stack(
        tensors,
        dim=0
    )

    return tensors


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_windows(
    model,
    output_classes,
    tensors,
    device
):

    # Only 8 windows, so they can be evaluated together.

    batch = tensors.to(
        device
    )

    with torch.inference_mode():

        logits = model(
            batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_classes = [
        output_classes[
            int(index)
        ]
        for index in predicted_indices
    ]

    del batch
    del logits

    return (
        probabilities,
        predicted_classes
    )


# ============================================================
# AGGREGATE WINDOWS
# ============================================================

def aggregate_predictions(
    probabilities,
    predicted_classes,
    output_classes
):

    # --------------------------------------------------------
    # 1. Majority vote.
    # --------------------------------------------------------

    counts = Counter(
        predicted_classes
    )

    (
        majority_class,
        majority_count
    ) = counts.most_common(
        1
    )[0]

    majority_percentage = (
        majority_count
        /
        len(predicted_classes)
        *
        100.0
    )

    fm_window_count = counts.get(
        FM_CLASS_NAME,
        0
    )

    fm_window_percentage = (
        fm_window_count
        /
        len(predicted_classes)
        *
        100.0
    )

    # --------------------------------------------------------
    # 2. Mean probability aggregation.
    # --------------------------------------------------------

    mean_probabilities = np.mean(
        probabilities,
        axis=0
    )

    top3_indices = np.argsort(
        mean_probabilities
    )[::-1][:3]

    aggregate_top3_classes = [
        output_classes[
            int(index)
        ]
        for index in top3_indices
    ]

    aggregate_top3_probabilities = [
        float(
            mean_probabilities[
                int(index)
            ]
            *
            100.0
        )
        for index in top3_indices
    ]

    fm_index = output_classes.index(
        FM_CLASS_NAME
    )

    mean_fm_probability = float(
        mean_probabilities[
            fm_index
        ]
        *
        100.0
    )

    return {
        "majority_class":
            majority_class,

        "majority_count":
            majority_count,

        "majority_percentage":
            majority_percentage,

        "fm_window_count":
            fm_window_count,

        "fm_window_percentage":
            fm_window_percentage,

        "aggregate_top1":
            aggregate_top3_classes[0],

        "aggregate_top1_probability":
            aggregate_top3_probabilities[0],

        "aggregate_top2":
            aggregate_top3_classes[1],

        "aggregate_top2_probability":
            aggregate_top3_probabilities[1],

        "aggregate_top3":
            aggregate_top3_classes[2],

        "aggregate_top3_probability":
            aggregate_top3_probabilities[2],

        "mean_fm_probability":
            mean_fm_probability,

        "fm_aggregate_top1":
            int(
                aggregate_top3_classes[0]
                ==
                FM_CLASS_NAME
            ),

        "fm_aggregate_top3":
            int(
                FM_CLASS_NAME
                in
                aggregate_top3_classes
            ),
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
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
# CREATE SUMMARY
# ============================================================

def create_summary(
    aggregate_rows
):

    summary = []

    for model_name in MODEL_PATHS:

        rows = [
            row
            for row in aggregate_rows
            if row["model"] == model_name
        ]

        total = len(
            rows
        )

        if total == 0:
            continue

        top1_correct = sum(
            int(
                row["fm_aggregate_top1"]
            )
            for row in rows
        )

        top3_correct = sum(
            int(
                row["fm_aggregate_top3"]
            )
            for row in rows
        )

        mean_window_rate = np.mean([
            float(
                row["fm_window_percentage"]
            )
            for row in rows
        ])

        mean_fm_probability = np.mean([
            float(
                row["mean_fm_probability"]
            )
            for row in rows
        ])

        summary.append({
            "model":
                model_name,

            "captures":
                total,

            "fm_top1_correct":
                top1_correct,

            "fm_top1_rate":
                top1_correct
                /
                total
                *
                100.0,

            "fm_top3_correct":
                top3_correct,

            "fm_top3_rate":
                top3_correct
                /
                total
                *
                100.0,

            "mean_fm_window_rate":
                float(
                    mean_window_rate
                ),

            "mean_fm_probability":
                float(
                    mean_fm_probability
                ),
        })

    return summary


# ============================================================
# PLOT:
# TOP-1 / TOP-3
# ============================================================

def create_top1_top3_plot(
    summary
):

    models_names = [
        row["model"]
        for row in summary
    ]

    top1 = [
        row["fm_top1_rate"]
        for row in summary
    ]

    top3 = [
        row["fm_top3_rate"]
        for row in summary
    ]

    x = np.arange(
        len(models_names)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars1 = ax.bar(
        x - width / 2,
        top1,
        width,
        label="Top-1"
    )

    bars2 = ax.bar(
        x + width / 2,
        top3,
        width,
        label="Top-3"
    )

    ax.set_ylabel(
        "FM recognition rate, %"
    )

    ax.set_title(
        "FM Recognition After "
        "8x I/Q Decimation"
    )

    ax.set_ylim(
        0,
        110
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        models_names,
        rotation=15,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bars in [
        bars1,
        bars2,
    ]:

        for bar in bars:

            value = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                +
                bar.get_width() / 2,

                value + 1,

                f"{value:.1f}%",

                ha="center",
                fontsize=8
            )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "resampled_fm_top1_top3.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] Plot saved: {path}"
    )


# ============================================================
# PLOT:
# MEAN FM PROBABILITY
# ============================================================

def create_probability_plot(
    summary
):

    model_names = [
        row["model"]
        for row in summary
    ]

    values = [
        row["mean_fm_probability"]
        for row in summary
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        model_names,
        values
    )

    ax.set_ylabel(
        "Mean FM probability, %"
    )

    ax.set_title(
        "Mean FM Probability "
        "After 8x I/Q Decimation"
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    max_value = max(
        values
    ) if values else 0

    offset = max(
        0.05,
        max_value * 0.02
    )

    for bar in bars:

        value = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            value + offset,

            f"{value:.3f}%",

            ha="center",
            fontsize=9
        )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "resampled_mean_fm_probability.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] Plot saved: {path}"
    )


# ============================================================
# PLOT:
# COMPARISON WITH EXPERIMENT A AND B
# ============================================================

def create_experiment_comparison_plot(
    summary
):

    # --------------------------------------------------------
    # Previously measured results.
    #
    # Experiment A:
    # one 1024-sample window
    #
    # Experiment B:
    # 64 original-rate windows + aggregation
    # --------------------------------------------------------

    experiment_a = {
        "ResNet18": 0.0,
        "MobileNetV3-Large": 0.0,
        "DenseNet121": 0.0,
        "ConvNeXt-Tiny": 0.0,
        "ViT-B/16": 0.0,
    }

    experiment_b = {
        "ResNet18": 16.67,
        "MobileNetV3-Large": 16.67,
        "DenseNet121": 0.0,
        "ConvNeXt-Tiny": 0.0,
        "ViT-B/16": 0.0,
    }

    model_names = [
        row["model"]
        for row in summary
    ]

    experiment_c = {
        row["model"]:
            row["fm_top1_rate"]
        for row in summary
    }

    a_values = [
        experiment_a[
            name
        ]
        for name in model_names
    ]

    b_values = [
        experiment_b[
            name
        ]
        for name in model_names
    ]

    c_values = [
        experiment_c[
            name
        ]
        for name in model_names
    ]

    x = np.arange(
        len(model_names)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        x - width,
        a_values,
        width,
        label="A: single 0.5 ms window"
    )

    ax.bar(
        x,
        b_values,
        width,
        label="B: 64-window aggregation"
    )

    ax.bar(
        x + width,
        c_values,
        width,
        label="C: 8x decimation"
    )

    ax.set_ylabel(
        "FM Top-1 recognition rate, %"
    )

    ax.set_title(
        "Comparison of Real FM "
        "Processing Strategies"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        model_names,
        rotation=15,
        ha="right"
    )

    ax.set_ylim(
        0,
        110
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "experiment_A_B_C_comparison.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[OK] Comparison plot saved: "
        f"{path}"
    )


# ============================================================
# TEXT REPORT
# ============================================================

def save_report(
    summary
):

    path = (
        OUTPUT_DIR /
        "experiment_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EXPERIMENT C - "
            "RTL-SDR FM RESAMPLING TEST\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            f"Original sample rate: "
            f"{ORIGINAL_SAMPLE_RATE / 1e6:.3f} MS/s\n"
        )

        file.write(
            f"Decimation factor: "
            f"{DECIMATION_FACTOR}\n"
        )

        file.write(
            f"Target sample rate: "
            f"{TARGET_SAMPLE_RATE / 1e3:.0f} kS/s\n"
        )

        file.write(
            f"Original samples per capture: "
            f"{ORIGINAL_CAPTURE_SAMPLES}\n"
        )

        file.write(
            f"Resampled samples per capture: "
            f"{EXPECTED_RESAMPLED_SAMPLES}\n"
        )

        file.write(
            f"Model window size: "
            f"{MODEL_WINDOW_SIZE}\n"
        )

        file.write(
            f"Windows per capture: "
            f"{NUM_WINDOWS}\n"
        )

        file.write(
            f"Physical duration represented "
            f"by one model window: "
            f"{MODEL_WINDOW_SIZE / TARGET_SAMPLE_RATE * 1000:.3f} ms\n"
        )

        file.write(
            "\nRESULTS\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )

        for row in summary:

            file.write(
                f"\n{row['model']}\n"
            )

            file.write(
                f"FM Top-1: "
                f"{row['fm_top1_correct']}/"
                f"{row['captures']} "
                f"= "
                f"{row['fm_top1_rate']:.2f}%\n"
            )

            file.write(
                f"FM Top-3: "
                f"{row['fm_top3_correct']}/"
                f"{row['captures']} "
                f"= "
                f"{row['fm_top3_rate']:.2f}%\n"
            )

            file.write(
                f"Mean FM window rate: "
                f"{row['mean_fm_window_rate']:.2f}%\n"
            )

            file.write(
                f"Mean FM probability: "
                f"{row['mean_fm_probability']:.4f}%\n"
            )

    print(
        f"[OK] Report saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "EXPERIMENT C - "
        "RTL-SDR FM RESAMPLING TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"\nInput I/Q directory:\n"
        f"{INPUT_IQ_DIR.resolve()}"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "\nSignal processing:"
    )

    print(
        f"  Original Fs: "
        f"{ORIGINAL_SAMPLE_RATE / 1e6:.3f} MS/s"
    )

    print(
        f"  Decimation factor: "
        f"{DECIMATION_FACTOR}"
    )

    print(
        f"  Target Fs: "
        f"{TARGET_SAMPLE_RATE / 1e3:.0f} kS/s"
    )

    print(
        f"  Original samples: "
        f"{ORIGINAL_CAPTURE_SAMPLES}"
    )

    print(
        f"  Resampled samples: "
        f"{EXPECTED_RESAMPLED_SAMPLES}"
    )

    print(
        f"  Windows after resampling: "
        f"{NUM_WINDOWS}"
    )

    original_window_ms = (
        MODEL_WINDOW_SIZE
        /
        ORIGINAL_SAMPLE_RATE
        *
        1000
    )

    resampled_window_ms = (
        MODEL_WINDOW_SIZE
        /
        TARGET_SAMPLE_RATE
        *
        1000
    )

    print(
        f"  Original 1024-sample duration: "
        f"{original_window_ms:.3f} ms"
    )

    print(
        f"  Resampled 1024-sample duration: "
        f"{resampled_window_ms:.3f} ms"
    )

    # ========================================================
    # VERIFY INPUT FILES
    # ========================================================

    print(
        "\nChecking saved I/Q captures..."
    )

    input_files = []

    for station in STATIONS:

        station_name = (
            station["name"]
        )

        for capture_number in range(
            1,
            CAPTURES_PER_STATION + 1
        ):

            filename = (
                f"{station_name}_"
                f"capture_{capture_number}_"
                f"65536.npy"
            )

            path = (
                INPUT_IQ_DIR /
                filename
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing I/Q file:\n"
                    f"{path}"
                )

            input_files.append(
                path
            )

            print(
                f"[OK] {path}"
            )

    print(
        f"\n[OK] All "
        f"{len(input_files)} "
        f"I/Q captures found."
    )

    # ========================================================
    # CHECK MODELS
    # ========================================================

    print(
        "\nChecking model files..."
    )

    for (
        model_name,
        model_path
    ) in MODEL_PATHS.items():

        if not Path(
            model_path
        ).exists():

            raise FileNotFoundError(
                f"{model_name} checkpoint "
                f"not found:\n"
                f"{model_path}"
            )

        print(
            f"[OK] {model_name}: "
            f"{model_path}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

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

    # ========================================================
    # LOAD MODELS
    # ========================================================

    loaded_models = {}

    for (
        model_name,
        model_path
    ) in MODEL_PATHS.items():

        loaded_models[
            model_name
        ] = load_model(
            model_name,
            model_path,
            device
        )

    print(
        "\n[OK] All five models loaded."
    )

    # ========================================================
    # RESULTS
    # ========================================================

    window_rows = []
    aggregate_rows = []

    # ========================================================
    # PROCESS EXISTING CAPTURES
    # ========================================================

    for (
        station_number,
        station
    ) in enumerate(
        STATIONS,
        start=1
    ):

        station_name = (
            station["name"]
        )

        frequency_mhz = (
            station["frequency_mhz"]
        )

        print(
            "\n"
            +
            "=" * 72
        )

        print(
            f"STATION "
            f"{station_number}/"
            f"{len(STATIONS)}: "
            f"{station_name}"
        )

        print(
            f"Frequency: "
            f"{frequency_mhz:.3f} MHz"
        )

        print(
            "=" * 72
        )

        for capture_number in range(
            1,
            CAPTURES_PER_STATION + 1
        ):

            print(
                f"\nCapture "
                f"{capture_number}/"
                f"{CAPTURES_PER_STATION}"
            )

            filename = (
                f"{station_name}_"
                f"capture_{capture_number}_"
                f"65536.npy"
            )

            input_path = (
                INPUT_IQ_DIR /
                filename
            )

            # ------------------------------------------------
            # Load EXACT SAME capture used in Experiment B.
            # ------------------------------------------------

            samples = np.load(
                input_path
            )

            samples = np.asarray(
                samples,
                dtype=np.complex64
            )

            if len(samples) != ORIGINAL_CAPTURE_SAMPLES:

                raise RuntimeError(
                    f"{input_path}: "
                    f"expected "
                    f"{ORIGINAL_CAPTURE_SAMPLES} "
                    f"samples, got "
                    f"{len(samples)}."
                )

            print(
                f"[OK] Loaded "
                f"{len(samples)} original samples."
            )

            # ------------------------------------------------
            # Resample.
            # ------------------------------------------------

            resampled = resample_iq(
                samples
            )

            print(
                f"[OK] Decimated "
                f"{len(samples)} -> "
                f"{len(resampled)} samples."
            )

            # ------------------------------------------------
            # Save the resampled signal separately.
            # ------------------------------------------------

            resampled_filename = (
                f"{station_name}_"
                f"capture_{capture_number}_"
                f"256ksps.npy"
            )

            resampled_path = (
                RESAMPLED_IQ_DIR /
                resampled_filename
            )

            np.save(
                resampled_path,
                resampled
            )

            print(
                f"[OK] Resampled I/Q saved: "
                f"{resampled_path}"
            )

            # ------------------------------------------------
            # 8192 -> 8 windows x 1024.
            # ------------------------------------------------

            tensors = create_window_tensors(
                resampled
            )

            # =================================================
            # FIVE MODELS
            # =================================================

            for (
                model_name,
                model_data
            ) in loaded_models.items():

                (
                    model,
                    output_classes
                ) = model_data

                print(
                    f"\nRunning {model_name} "
                    f"on {NUM_WINDOWS} "
                    f"resampled windows..."
                )

                (
                    probabilities,
                    predicted_classes
                ) = predict_windows(
                    model,
                    output_classes,
                    tensors,
                    device
                )

                fm_index = (
                    output_classes.index(
                        FM_CLASS_NAME
                    )
                )

                # --------------------------------------------
                # Individual window predictions.
                # --------------------------------------------

                for window_index in range(
                    NUM_WINDOWS
                ):

                    predicted_class = (
                        predicted_classes[
                            window_index
                        ]
                    )

                    predicted_index = (
                        output_classes.index(
                            predicted_class
                        )
                    )

                    top1_probability = (
                        probabilities[
                            window_index,
                            predicted_index
                        ]
                        *
                        100.0
                    )

                    fm_probability = (
                        probabilities[
                            window_index,
                            fm_index
                        ]
                        *
                        100.0
                    )

                    window_rows.append({
                        "station":
                            station_name,

                        "frequency_mhz":
                            frequency_mhz,

                        "capture":
                            capture_number,

                        "model":
                            model_name,

                        "window":
                            window_index + 1,

                        "resampled_sample_start":
                            window_index
                            *
                            MODEL_WINDOW_SIZE,

                        "resampled_sample_end":
                            (
                                window_index + 1
                            )
                            *
                            MODEL_WINDOW_SIZE
                            -
                            1,

                        "physical_start_ms":
                            (
                                window_index
                                *
                                MODEL_WINDOW_SIZE
                                /
                                TARGET_SAMPLE_RATE
                                *
                                1000.0
                            ),

                        "physical_end_ms":
                            (
                                (
                                    window_index + 1
                                )
                                *
                                MODEL_WINDOW_SIZE
                                /
                                TARGET_SAMPLE_RATE
                                *
                                1000.0
                            ),

                        "top1":
                            predicted_class,

                        "top1_probability":
                            float(
                                top1_probability
                            ),

                        "fm_probability":
                            float(
                                fm_probability
                            ),

                        "is_fm":
                            int(
                                predicted_class
                                ==
                                FM_CLASS_NAME
                            ),
                    })

                # --------------------------------------------
                # Aggregate 8 windows.
                # --------------------------------------------

                aggregate = (
                    aggregate_predictions(
                        probabilities,
                        predicted_classes,
                        output_classes
                    )
                )

                aggregate_rows.append({
                    "station":
                        station_name,

                    "frequency_mhz":
                        frequency_mhz,

                    "capture":
                        capture_number,

                    "model":
                        model_name,

                    "decimation_factor":
                        DECIMATION_FACTOR,

                    "target_sample_rate":
                        TARGET_SAMPLE_RATE,

                    "windows":
                        NUM_WINDOWS,

                    **aggregate,
                })

                print(
                    f"  Majority class: "
                    f"{aggregate['majority_class']} "
                    f"("
                    f"{aggregate['majority_count']}/"
                    f"{NUM_WINDOWS}, "
                    f"{aggregate['majority_percentage']:.2f}%"
                    f")"
                )

                print(
                    f"  Windows classified as FM: "
                    f"{aggregate['fm_window_count']}/"
                    f"{NUM_WINDOWS} "
                    f"("
                    f"{aggregate['fm_window_percentage']:.2f}%"
                    f")"
                )

                print(
                    f"  Aggregate Top-1: "
                    f"{aggregate['aggregate_top1']} "
                    f"("
                    f"{aggregate['aggregate_top1_probability']:.2f}%"
                    f")"
                )

                print(
                    f"  Mean FM probability: "
                    f"{aggregate['mean_fm_probability']:.4f}%"
                )

                print(
                    f"  FM in aggregate Top-3: "
                    f"{'YES' if aggregate['fm_aggregate_top3'] else 'NO'}"
                )

            # ------------------------------------------------
            # Save progress after every capture.
            # ------------------------------------------------

            save_csv(
                TABLES_DIR /
                "window_predictions.csv",
                window_rows
            )

            save_csv(
                TABLES_DIR /
                "aggregate_results.csv",
                aggregate_rows
            )

            print(
                "[OK] Partial results saved."
            )

            del samples
            del resampled
            del tensors

            gc.collect()

            if device.type == "cuda":

                torch.cuda.empty_cache()

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = create_summary(
        aggregate_rows
    )

    save_csv(
        TABLES_DIR /
        "model_summary.csv",
        summary
    )

    # ========================================================
    # GRAPHICS
    # ========================================================

    create_top1_top3_plot(
        summary
    )

    create_probability_plot(
        summary
    )

    create_experiment_comparison_plot(
        summary
    )

    save_report(
        summary
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "FINAL EXPERIMENT C RESULTS"
    )

    print(
        "=" * 72
    )

    for row in summary:

        print(
            f"\n{row['model']}"
        )

        print(
            f"  FM Top-1: "
            f"{row['fm_top1_correct']}/"
            f"{row['captures']} "
            f"= "
            f"{row['fm_top1_rate']:.2f}%"
        )

        print(
            f"  FM Top-3: "
            f"{row['fm_top3_correct']}/"
            f"{row['captures']} "
            f"= "
            f"{row['fm_top3_rate']:.2f}%"
        )

        print(
            f"  Mean windows classified as FM: "
            f"{row['mean_fm_window_rate']:.2f}%"
        )

        print(
            f"  Mean FM probability: "
            f"{row['mean_fm_probability']:.4f}%"
        )

    # ========================================================
    # COUNTS
    # ========================================================

    expected_captures = (
        len(STATIONS)
        *
        CAPTURES_PER_STATION
    )

    expected_aggregate_rows = (
        expected_captures
        *
        len(MODEL_PATHS)
    )

    expected_window_rows = (
        expected_captures
        *
        NUM_WINDOWS
        *
        len(MODEL_PATHS)
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 72
    )

    print(
        f"\nInput captures: "
        f"{expected_captures}"
    )

    print(
        f"Aggregate prediction rows: "
        f"{len(aggregate_rows)} "
        f"(expected "
        f"{expected_aggregate_rows})"
    )

    print(
        f"Window prediction rows: "
        f"{len(window_rows)} "
        f"(expected "
        f"{expected_window_rows})"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "\nGenerated:"
    )

    print(
        "  resampled_iq/"
    )

    print(
        "  tables/window_predictions.csv"
    )

    print(
        "  tables/aggregate_results.csv"
    )

    print(
        "  tables/model_summary.csv"
    )

    print(
        "  plots/resampled_fm_top1_top3.png"
    )

    print(
        "  plots/resampled_mean_fm_probability.png"
    )

    print(
        "  plots/experiment_A_B_C_comparison.png"
    )

    print(
        "  experiment_report.txt"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()