import csv
import gc
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from torchvision import models, transforms
from rtlsdr import RtlSdr


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Separate directory. The previous experiment is NOT overwritten.
OUTPUT_DIR = Path("fm_multiwindow_experiment")

TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"
IQ_DIR = OUTPUT_DIR / "iq_samples"

for directory in [
    OUTPUT_DIR,
    TABLES_DIR,
    PLOTS_DIR,
    IQ_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# SDR SETTINGS
# ============================================================

SAMPLE_RATE = 2.048e6
GAIN = 40

# One complete capture = 65536 complex I/Q samples.
NUM_SAMPLES = 65536

# Model input remains EXACTLY the same as during training.
WINDOW_SIZE = 1024

# 65536 / 1024 = 64 windows.
NUM_WINDOWS = NUM_SAMPLES // WINDOW_SIZE

TUNE_DELAY = 1.0
CAPTURE_DELAY = 1.0

FLUSH_BUFFERS = 10
FLUSH_SAMPLES = 65536


# ============================================================
# SPECTROGRAM SETTINGS
# ============================================================

IMAGE_SIZE = 224

NFFT = 64
NOVERLAP = 32
SPECTROGRAM_FS = 1000


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NUM_CLASSES = 24
FM_CLASS_NAME = "FM"

CAPTURES_PER_STATION = 2


STATIONS = [
    {
        "name": "Army_FM",
        "frequency": 94.6e6,
    },
    {
        "name": "Radio_NV",
        "frequency": 96.0e6,
    },
    {
        "name": "Kyiv_FM",
        "frequency": 98.0e6,
    },
    {
        "name": "Radio_ROKS",
        "frequency": 103.6e6,
    },
    {
        "name": "Radio_Jazz",
        "frequency": 104.6e6,
    },
    {
        "name": "Kiss_FM",
        "frequency": 106.5e6,
    },
]


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
# ORIGINAL RADIOML CLASS LIST
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
    # Restore exact output class mapping.
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
            x is None
            for x in output_classes
        ):

            raise ValueError(
                f"{model_name}: incomplete class_to_idx."
            )

    elif checkpoint_classes is not None:

        output_classes = list(
            checkpoint_classes
        )

    else:

        # ImageFolder uses alphabetical class ordering.
        output_classes = sorted(
            CLASSES
        )

    if len(output_classes) != NUM_CLASSES:

        raise ValueError(
            f"{model_name}: incorrect number of classes."
        )

    if FM_CLASS_NAME not in output_classes:

        raise ValueError(
            f"{model_name}: FM class is missing."
        )

    print(
        f"[OK] {model_name} loaded."
    )

    return model, output_classes


# ============================================================
# IQ WINDOW -> SPECTROGRAM TENSOR
# ============================================================

def iq_to_tensor(samples):

    samples = np.asarray(
        samples
    )

    if len(samples) != WINDOW_SIZE:

        raise ValueError(
            f"Expected {WINDOW_SIZE} samples, "
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
            f"Incorrect image size: {image.size}"
        )

    tensor = IMAGE_TRANSFORM(
        image
    )

    return tensor


# ============================================================
# CREATE 64 WINDOW TENSORS
# ============================================================

def create_window_tensors(samples):

    required_samples = (
        NUM_WINDOWS
        *
        WINDOW_SIZE
    )

    samples = np.asarray(
        samples[:required_samples]
    )

    tensors = []

    print(
        f"Creating {NUM_WINDOWS} "
        f"spectrogram windows..."
    )

    for window_index in range(
        NUM_WINDOWS
    ):

        start = (
            window_index
            *
            WINDOW_SIZE
        )

        end = (
            start
            +
            WINDOW_SIZE
        )

        window = samples[
            start:end
        ]

        tensor = iq_to_tensor(
            window
        )

        tensors.append(
            tensor
        )

        if (
            (window_index + 1) % 8 == 0
            or
            window_index + 1 == NUM_WINDOWS
        ):

            print(
                f"  Spectrograms: "
                f"{window_index + 1}/"
                f"{NUM_WINDOWS}"
            )

    tensors = torch.stack(
        tensors,
        dim=0
    )

    print(
        f"[OK] {NUM_WINDOWS} "
        f"window tensors created."
    )

    return tensors


# ============================================================
# CAPTURE RTL-SDR
# ============================================================

def capture_signal(
    sdr,
    frequency
):

    print(
        f"\nTuning to "
        f"{frequency / 1e6:.3f} MHz..."
    )

    sdr.center_freq = frequency

    time.sleep(
        TUNE_DELAY
    )

    # Flush samples after retuning.
    for _ in range(
        FLUSH_BUFFERS
    ):

        _ = sdr.read_samples(
            FLUSH_SAMPLES
        )

    time.sleep(
        CAPTURE_DELAY
    )

    samples = sdr.read_samples(
        NUM_SAMPLES
    )

    samples = np.asarray(
        samples,
        dtype=np.complex64
    )

    if len(samples) < NUM_SAMPLES:

        raise RuntimeError(
            f"Expected {NUM_SAMPLES} samples, "
            f"received {len(samples)}."
        )

    print(
        f"[OK] Captured "
        f"{len(samples)} I/Q samples."
    )

    return samples


# ============================================================
# PREDICT ALL WINDOWS
# ============================================================

def predict_windows(
    model,
    output_classes,
    tensors,
    device
):

    # Process in small batches to avoid GPU memory problems.
    batch_size = 8

    all_probabilities = []

    with torch.inference_mode():

        for start in range(
            0,
            len(tensors),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(tensors)
            )

            batch = tensors[
                start:end
            ].to(
                device
            )

            logits = model(
                batch
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            all_probabilities.append(
                probabilities.cpu()
            )

            del batch
            del logits
            del probabilities

    probabilities = torch.cat(
        all_probabilities,
        dim=0
    )

    predicted_indices = torch.argmax(
        probabilities,
        dim=1
    ).numpy()

    predicted_classes = [
        output_classes[
            int(index)
        ]
        for index in predicted_indices
    ]

    return (
        probabilities.numpy(),
        predicted_classes
    )


# ============================================================
# AGGREGATE 64 WINDOWS
# ============================================================

def aggregate_predictions(
    probabilities,
    predicted_classes,
    output_classes
):

    # --------------------------------------------------------
    # METHOD 1:
    # Majority vote based on Top-1 class of each window.
    # --------------------------------------------------------

    counts = Counter(
        predicted_classes
    )

    majority_class, majority_count = (
        counts.most_common(1)[0]
    )

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
    # METHOD 2:
    # Average Softmax probability over all 64 windows.
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
# SAVE DETAILED WINDOW RESULTS
# ============================================================

def save_window_results(
    rows
):

    path = (
        TABLES_DIR /
        "window_predictions.csv"
    )

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

    print(
        f"[OK] Window predictions saved: {path}"
    )


# ============================================================
# SAVE AGGREGATED RESULTS
# ============================================================

def save_aggregate_results(
    rows
):

    path = (
        TABLES_DIR /
        "aggregate_results.csv"
    )

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

    print(
        f"[OK] Aggregate results saved: {path}"
    )


# ============================================================
# CREATE MODEL SUMMARY
# ============================================================

def create_model_summary(
    aggregate_rows
):

    summary = []

    for model_name in MODEL_PATHS:

        selected = [
            row
            for row in aggregate_rows
            if row["model"] == model_name
        ]

        total = len(
            selected
        )

        if total == 0:
            continue

        top1_correct = sum(
            int(
                row["fm_aggregate_top1"]
            )
            for row in selected
        )

        top3_correct = sum(
            int(
                row["fm_aggregate_top3"]
            )
            for row in selected
        )

        average_fm_windows = np.mean([
            float(
                row["fm_window_percentage"]
            )
            for row in selected
        ])

        average_fm_probability = np.mean([
            float(
                row["mean_fm_probability"]
            )
            for row in selected
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
                average_fm_windows,

            "mean_fm_probability":
                average_fm_probability,
        })

    return summary


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    summary
):

    path = (
        TABLES_DIR /
        "model_summary.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            summary
        )

    print(
        f"[OK] Model summary saved: {path}"
    )


# ============================================================
# GRAPH 1:
# AGGREGATED FM TOP-1 / TOP-3
# ============================================================

def create_top1_top3_plot(
    summary
):

    model_names = [
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
        len(model_names)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars1 = ax.bar(
        x - width / 2,
        top1,
        width,
        label="Aggregated Top-1"
    )

    bars2 = ax.bar(
        x + width / 2,
        top3,
        width,
        label="Aggregated Top-3"
    )

    ax.set_ylabel(
        "FM recognition rate, %"
    )

    ax.set_title(
        "RTL-SDR FM Recognition "
        "Using 64-Window Aggregation"
    )

    ax.set_ylim(
        0,
        110
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        model_names,
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
        "multiwindow_fm_top1_top3.png"
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
# GRAPH 2:
# PERCENTAGE OF WINDOWS CLASSIFIED AS FM
# ============================================================

def create_fm_window_plot(
    summary
):

    model_names = [
        row["model"]
        for row in summary
    ]

    values = [
        row["mean_fm_window_rate"]
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
        "Windows classified as FM, %"
    )

    ax.set_title(
        "Mean Percentage of 1024-Sample "
        "Windows Classified as FM"
    )

    ax.set_ylim(
        0,
        max(
            10,
            max(values) * 1.2
        )
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar in bars:

        value = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            value
            +
            max(
                0.2,
                max(values) * 0.02
            ),

            f"{value:.2f}%",

            ha="center",
            fontsize=9
        )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "fm_window_percentage.png"
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
# GRAPH 3:
# MEAN FM SOFTMAX PROBABILITY
# ============================================================

def create_fm_probability_plot(
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
        "Mean FM Softmax Probability "
        "Across RTL-SDR Windows"
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar in bars:

        value = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            value
            +
            max(
                0.05,
                max(values) * 0.02
            ),

            f"{value:.3f}%",

            ha="center",
            fontsize=9
        )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "mean_fm_probability.png"
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
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "RTL-SDR FM MULTI-WINDOW EXPERIMENT"
    )

    print(
        "=" * 72
    )

    print(
        f"\nSample rate: "
        f"{SAMPLE_RATE / 1e6:.3f} MS/s"
    )

    print(
        f"Samples per capture: {NUM_SAMPLES}"
    )

    print(
        f"Window size: {WINDOW_SIZE}"
    )

    print(
        f"Windows per capture: {NUM_WINDOWS}"
    )

    window_duration_ms = (
        WINDOW_SIZE
        /
        SAMPLE_RATE
        *
        1000
    )

    capture_duration_ms = (
        NUM_SAMPLES
        /
        SAMPLE_RATE
        *
        1000
    )

    print(
        f"Single window duration: "
        f"{window_duration_ms:.3f} ms"
    )

    print(
        f"Total capture duration: "
        f"{capture_duration_ms:.3f} ms"
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
    # CHECK MODEL FILES
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
                f"{model_name}: "
                f"{model_path}"
            )

        print(
            f"[OK] {model_name}: "
            f"{model_path}"
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
    # RESULT STORAGE
    # ========================================================

    window_rows = []
    aggregate_rows = []

    # ========================================================
    # OPEN RTL-SDR
    # ========================================================

    print(
        "\nOpening RTL-SDR..."
    )

    sdr = RtlSdr()

    sdr.sample_rate = SAMPLE_RATE
    sdr.gain = GAIN

    print(
        "[OK] RTL-SDR opened."
    )

    try:

        # ====================================================
        # STATIONS
        # ====================================================

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

            frequency = (
                station["frequency"]
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
                f"{frequency / 1e6:.3f} MHz"
            )

            print(
                "=" * 72
            )

            # =================================================
            # CAPTURES
            # =================================================

            for capture_number in range(
                1,
                CAPTURES_PER_STATION + 1
            ):

                print(
                    f"\nCapture "
                    f"{capture_number}/"
                    f"{CAPTURES_PER_STATION}"
                )

                # ---------------------------------------------
                # Capture 65536 samples.
                # ---------------------------------------------

                samples = capture_signal(
                    sdr,
                    frequency
                )

                # ---------------------------------------------
                # Save complete raw capture.
                # ---------------------------------------------

                iq_filename = (
                    f"{station_name}_"
                    f"capture_{capture_number}_"
                    f"65536.npy"
                )

                iq_path = (
                    IQ_DIR /
                    iq_filename
                )

                np.save(
                    iq_path,
                    samples
                )

                print(
                    f"[OK] Full I/Q capture saved: "
                    f"{iq_path}"
                )

                # ---------------------------------------------
                # Create EXACT SAME 1024-sample representation
                # for each of the 64 windows.
                # ---------------------------------------------

                tensors = create_window_tensors(
                    samples
                )

                # =============================================
                # ALL FIVE MODELS
                # =============================================

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
                        f"on {NUM_WINDOWS} windows..."
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

                    # -----------------------------------------
                    # Save each individual window prediction.
                    # -----------------------------------------

                    for window_index in range(
                        NUM_WINDOWS
                    ):

                        prediction = (
                            predicted_classes[
                                window_index
                            ]
                        )

                        class_index = (
                            output_classes.index(
                                prediction
                            )
                        )

                        top1_probability = (
                            probabilities[
                                window_index,
                                class_index
                            ]
                            *
                            100.0
                        )

                        fm_index = (
                            output_classes.index(
                                FM_CLASS_NAME
                            )
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
                                frequency / 1e6,

                            "capture":
                                capture_number,

                            "model":
                                model_name,

                            "window":
                                window_index + 1,

                            "sample_start":
                                window_index
                                *
                                WINDOW_SIZE,

                            "sample_end":
                                (
                                    window_index + 1
                                )
                                *
                                WINDOW_SIZE
                                -
                                1,

                            "top1":
                                prediction,

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
                                    prediction
                                    ==
                                    FM_CLASS_NAME
                                ),
                        })

                    # -----------------------------------------
                    # Aggregate 64 predictions.
                    # -----------------------------------------

                    aggregate = (
                        aggregate_predictions(
                            probabilities,
                            predicted_classes,
                            output_classes
                        )
                    )

                    aggregate_row = {
                        "station":
                            station_name,

                        "frequency_mhz":
                            frequency / 1e6,

                        "capture":
                            capture_number,

                        "model":
                            model_name,

                        "windows":
                            NUM_WINDOWS,

                        **aggregate,
                    }

                    aggregate_rows.append(
                        aggregate_row
                    )

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
                        f"  Mean-probability Top-1: "
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

                # ---------------------------------------------
                # Save partial results after each capture.
                # ---------------------------------------------

                save_window_results(
                    window_rows
                )

                save_aggregate_results(
                    aggregate_rows
                )

                del samples
                del tensors

                gc.collect()

                if device.type == "cuda":

                    torch.cuda.empty_cache()

    finally:

        print(
            "\nClosing RTL-SDR..."
        )

        sdr.close()

        print(
            "[OK] RTL-SDR closed."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = create_model_summary(
        aggregate_rows
    )

    save_summary(
        summary
    )

    # ========================================================
    # GRAPHICAL MATERIALS
    # ========================================================

    create_top1_top3_plot(
        summary
    )

    create_fm_window_plot(
        summary
    )

    create_fm_probability_plot(
        summary
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "FINAL MULTI-WINDOW RESULTS"
    )

    print(
        "=" * 72
    )

    for row in summary:

        print(
            f"\n{row['model']}"
        )

        print(
            f"  Aggregate FM Top-1: "
            f"{row['fm_top1_correct']}/"
            f"{row['captures']} "
            f"= "
            f"{row['fm_top1_rate']:.2f}%"
        )

        print(
            f"  Aggregate FM Top-3: "
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
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "\nExperiment design:"
    )

    print(
        f"  Stations: {len(STATIONS)}"
    )

    print(
        f"  Captures per station: "
        f"{CAPTURES_PER_STATION}"
    )

    print(
        f"  Total captures: "
        f"{len(STATIONS) * CAPTURES_PER_STATION}"
    )

    print(
        f"  Samples per capture: "
        f"{NUM_SAMPLES}"
    )

    print(
        f"  Windows per capture: "
        f"{NUM_WINDOWS}"
    )

    print(
        f"  Samples per window: "
        f"{WINDOW_SIZE}"
    )

    print(
        f"  Total window predictions: "
        f"{len(window_rows)}"
    )

    print(
        "\nGenerated files:"
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
        "  plots/multiwindow_fm_top1_top3.png"
    )

    print(
        "  plots/fm_window_percentage.png"
    )

    print(
        "  plots/mean_fm_probability.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()