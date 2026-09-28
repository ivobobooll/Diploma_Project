import csv
import gc
import time
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

OUTPUT_DIR = Path("fm_5models_experiment")

SPECTROGRAM_DIR = OUTPUT_DIR / "spectrograms"
IQ_DIR = OUTPUT_DIR / "iq_samples"
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# SDR SETTINGS
# ============================================================

SAMPLE_RATE = 2.048e6
GAIN = 40

# Capture a larger block from RTL-SDR.
NUM_SAMPLES = 65536

# Exactly 1024 samples are passed to the neural networks.
MODEL_INPUT_SAMPLES = 1024

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

# Must match training preprocessing.
SPECTROGRAM_FS = 1000


# ============================================================
# CLASSES
# ============================================================

NUM_CLASSES = 24

# IMPORTANT FIX
FM_CLASS_NAME = "FM"


# ============================================================
# FM STATIONS
# ============================================================

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

    directories = [
        OUTPUT_DIR,
        SPECTROGRAM_DIR,
        IQ_DIR,
        TABLES_DIR,
        PLOTS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CHECK MODEL FILES
# ============================================================

def check_model_files():

    print("\nChecking model files...")

    for model_name, model_path in MODEL_PATHS.items():

        path = Path(model_path)

        if not path.exists():

            raise FileNotFoundError(
                f"{model_name} checkpoint not found:\n"
                f"{path}"
            )

        print(
            f"[OK] {model_name}: {path}"
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
    # Restore exact output class mapping from checkpoint.
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
                f"Incomplete class_to_idx "
                f"in {model_name} checkpoint."
            )

    elif checkpoint_classes is not None:

        output_classes = list(
            checkpoint_classes
        )

    else:

        # Training used ImageFolder.
        output_classes = sorted(
            CLASSES
        )

    if len(output_classes) != NUM_CLASSES:

        raise ValueError(
            f"{model_name}: expected "
            f"{NUM_CLASSES} classes, "
            f"got {len(output_classes)}."
        )

    if FM_CLASS_NAME not in output_classes:

        raise ValueError(
            f"{model_name}: FM class not found "
            f"in output class mapping."
        )

    print(
        f"[OK] {model_name} loaded."
    )

    return model, output_classes


# ============================================================
# IQ -> EXACT 224x224 SPECTROGRAM
# ============================================================

def iq_to_spectrogram(samples):

    samples = np.asarray(
        samples[:MODEL_INPUT_SAMPLES]
    )

    if len(samples) != MODEL_INPUT_SAMPLES:

        raise RuntimeError(
            f"Expected {MODEL_INPUT_SAMPLES} samples, "
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
            f"Spectrogram size is {image.size}; "
            f"expected "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    tensor = IMAGE_TRANSFORM(
        image
    )

    return image, tensor


# ============================================================
# SAVE SPECTROGRAM
# ============================================================

def save_spectrogram(
    image,
    station_name,
    capture_number
):

    filename = (
        f"{station_name}_"
        f"capture_{capture_number}.png"
    )

    path = (
        SPECTROGRAM_DIR
        /
        filename
    )

    image.save(
        path
    )

    return path


# ============================================================
# CAPTURE SIGNAL
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

    # --------------------------------------------------------
    # Flush old samples after changing frequency.
    # --------------------------------------------------------

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

    if len(samples) < MODEL_INPUT_SAMPLES:

        raise RuntimeError(
            "RTL-SDR returned too few samples."
        )

    print(
        f"[OK] Captured "
        f"{len(samples)} I/Q samples."
    )

    return samples


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(
    model,
    output_classes,
    tensor,
    device
):

    batch = (
        tensor
        .unsqueeze(0)
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

        (
            top_probabilities,
            top_indices
        ) = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    top_probabilities = (
        top_probabilities[0]
        .detach()
        .cpu()
        .numpy()
        *
        100.0
    )

    top_indices = (
        top_indices[0]
        .detach()
        .cpu()
        .numpy()
    )

    top_classes = [
        output_classes[
            int(index)
        ]
        for index in top_indices
    ]

    del batch
    del outputs
    del probabilities

    return (
        top_classes,
        top_probabilities
    )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

def save_results_csv(rows):

    if not rows:
        return

    path = (
        TABLES_DIR /
        "fm_experiment_results.csv"
    )

    fieldnames = [
        "station",
        "frequency_mhz",
        "capture",
        "model",
        "top1",
        "top1_probability",
        "top2",
        "top2_probability",
        "top3",
        "top3_probability",
        "fm_top1_correct",
        "fm_in_top3",
        "spectrogram",
        "iq_file",
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
        f"[OK] Results CSV saved: {path}"
    )


# ============================================================
# CALCULATE MODEL SUMMARY
# ============================================================

def calculate_summary(rows):

    summary = []

    for model_name in MODEL_PATHS:

        model_rows = [
            row
            for row in rows
            if row["model"] == model_name
        ]

        total = len(
            model_rows
        )

        if total == 0:
            continue

        top1_correct = sum(
            int(
                row["fm_top1_correct"]
            )
            for row in model_rows
        )

        top3_correct = sum(
            int(
                row["fm_in_top3"]
            )
            for row in model_rows
        )

        summary.append({
            "Model":
                model_name,

            "Captures":
                total,

            "FM Top-1 Correct":
                top1_correct,

            "FM Top-1 Rate":
                top1_correct / total,

            "FM Top-3 Correct":
                top3_correct,

            "FM Top-3 Rate":
                top3_correct / total,
        })

    return summary


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary_csv(summary):

    if not summary:
        return

    path = (
        TABLES_DIR /
        "fm_model_summary.csv"
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
        f"[OK] Summary CSV saved: {path}"
    )


# ============================================================
# SUMMARY TABLE — 300 DPI
# ============================================================

def create_summary_table(summary):

    columns = [
        "Model",
        "FM Top-1",
        "FM Top-3",
    ]

    data = []

    for row in summary:

        data.append([
            row["Model"],

            (
                f"{row['FM Top-1 Correct']}/"
                f"{row['Captures']} "
                f"({row['FM Top-1 Rate'] * 100:.1f}%)"
            ),

            (
                f"{row['FM Top-3 Correct']}/"
                f"{row['Captures']} "
                f"({row['FM Top-3 Rate'] * 100:.1f}%)"
            ),
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
        "RTL-SDR FM Recognition Results",
        fontsize=15,
        pad=20
    )

    fig.tight_layout()

    path = (
        TABLES_DIR /
        "fm_model_summary_table.png"
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
        f"[OK] Summary table saved: {path}"
    )


# ============================================================
# TOP-1 / TOP-3 GRAPH — 300 DPI
# ============================================================

def create_fm_rate_plot(summary):

    model_names = [
        row["Model"]
        for row in summary
    ]

    top1 = [
        row["FM Top-1 Rate"] * 100
        for row in summary
    ]

    top3 = [
        row["FM Top-3 Rate"] * 100
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

    ax.set_ylim(
        0,
        110
    )

    ax.set_title(
        "FM Recognition on Real RTL-SDR Signals"
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
                va="bottom",
                fontsize=8
            )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "rtl_sdr_fm_top1_top3.png"
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
        f"[OK] FM Top-1/Top-3 plot saved: "
        f"{path}"
    )


# ============================================================
# PER-STATION GRAPH — 300 DPI
# ============================================================

def create_station_plot(rows):

    station_names = [
        station["name"]
        for station in STATIONS
    ]

    model_names = list(
        MODEL_PATHS.keys()
    )

    data = np.zeros(
        (
            len(model_names),
            len(station_names)
        ),
        dtype=float
    )

    for (
        model_index,
        model_name
    ) in enumerate(
        model_names
    ):

        for (
            station_index,
            station_name
        ) in enumerate(
            station_names
        ):

            selected_rows = [
                row
                for row in rows
                if (
                    row["model"]
                    ==
                    model_name
                    and
                    row["station"]
                    ==
                    station_name
                )
            ]

            if selected_rows:

                correct = sum(
                    int(
                        row[
                            "fm_top1_correct"
                        ]
                    )
                    for row in selected_rows
                )

                data[
                    model_index,
                    station_index
                ] = (
                    correct
                    /
                    len(selected_rows)
                    *
                    100
                )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    image = ax.imshow(
        data,
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_xticks(
        np.arange(
            len(station_names)
        )
    )

    ax.set_xticklabels(
        station_names,
        rotation=30,
        ha="right"
    )

    ax.set_yticks(
        np.arange(
            len(model_names)
        )
    )

    ax.set_yticklabels(
        model_names
    )

    ax.set_xlabel(
        "FM station"
    )

    ax.set_ylabel(
        "Model"
    )

    ax.set_title(
        "FM Top-1 Recognition by Station"
    )

    for i in range(
        len(model_names)
    ):

        for j in range(
            len(station_names)
        ):

            ax.text(
                j,
                i,
                f"{data[i, j]:.0f}%",
                ha="center",
                va="center"
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "FM Top-1 recognition rate, %"
    )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "fm_recognition_by_station.png"
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
        f"[OK] Per-station plot saved: "
        f"{path}"
    )


# ============================================================
# RADIOML FM VS REAL RTL-SDR FM — 300 DPI
# ============================================================

def create_radioml_vs_real_plot(
    summary
):

    # Independent FM-only RadioML experiment:
    # 500 previously unseen FM samples.
    #
    # ResNet18          500/500
    # MobileNetV3       500/500
    # DenseNet121       496/500
    # ConvNeXt-Tiny     500/500
    # ViT-B/16          500/500

    radioml_fm_top1 = {
        "ResNet18": 100.0,
        "MobileNetV3-Large": 100.0,
        "DenseNet121": 99.2,
        "ConvNeXt-Tiny": 100.0,
        "ViT-B/16": 100.0,
    }

    model_names = [
        row["Model"]
        for row in summary
    ]

    radioml_values = [
        radioml_fm_top1[
            model_name
        ]
        for model_name in model_names
    ]

    real_values = [
        row["FM Top-1 Rate"] * 100
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
        radioml_values,
        width,
        label="RadioML FM"
    )

    bars2 = ax.bar(
        x + width / 2,
        real_values,
        width,
        label="RTL-SDR FM"
    )

    ax.set_ylabel(
        "FM Top-1 recognition rate, %"
    )

    ax.set_ylim(
        0,
        110
    )

    ax.set_title(
        "FM Recognition: "
        "RadioML vs Real RTL-SDR"
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
                va="bottom",
                fontsize=8
            )

    fig.tight_layout()

    path = (
        PLOTS_DIR /
        "radioml_vs_rtlsdr_fm.png"
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
        "[OK] RadioML vs RTL-SDR "
        f"plot saved: {path}"
    )


# ============================================================
# TEXT REPORT
# ============================================================

def save_text_report(
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
            "RTL-SDR FM EXPERIMENT\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            f"Sample rate: "
            f"{SAMPLE_RATE / 1e6:.3f} MS/s\n"
        )

        file.write(
            f"Gain: {GAIN} dB\n"
        )

        file.write(
            f"Captured samples: "
            f"{NUM_SAMPLES}\n"
        )

        file.write(
            f"Model input samples: "
            f"{MODEL_INPUT_SAMPLES}\n"
        )

        file.write(
            f"Spectrogram: "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        file.write(
            f"NFFT: {NFFT}\n"
        )

        file.write(
            f"noverlap: {NOVERLAP}\n"
        )

        file.write(
            f"Spectrogram Fs: "
            f"{SPECTROGRAM_FS}\n"
        )

        file.write(
            f"Captures per station: "
            f"{CAPTURES_PER_STATION}\n\n"
        )

        file.write(
            "STATIONS\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )

        for station in STATIONS:

            file.write(
                f"{station['name']}: "
                f"{station['frequency'] / 1e6:.3f} MHz\n"
            )

        file.write(
            "\nMODEL SUMMARY\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )

        for row in summary:

            file.write(
                f"{row['Model']}: "
                f"Top-1 "
                f"{row['FM Top-1 Correct']}/"
                f"{row['Captures']} "
                f"= "
                f"{row['FM Top-1 Rate'] * 100:.2f}%, "
                f"Top-3 "
                f"{row['FM Top-3 Correct']}/"
                f"{row['Captures']} "
                f"= "
                f"{row['FM Top-3 Rate'] * 100:.2f}%\n"
            )

    print(
        f"[OK] Text report saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "RTL-SDR FM EXPERIMENT — "
        "5 NEURAL NETWORK MODELS"
    )

    print(
        "=" * 70
    )

    create_directories()

    check_model_files()

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
    # LOAD ALL MODELS
    # ========================================================

    loaded_models = {}

    print(
        "\nLoading all models..."
    )

    for (
        model_name,
        model_path
    ) in MODEL_PATHS.items():

        (
            model,
            output_classes
        ) = load_model(
            model_name,
            model_path,
            device
        )

        loaded_models[
            model_name
        ] = (
            model,
            output_classes
        )

    print(
        "\n[OK] All five models loaded."
    )

    # ========================================================
    # OPEN RTL-SDR
    # ========================================================

    print(
        "\nOpening RTL-SDR..."
    )

    sdr = RtlSdr()

    sdr.sample_rate = (
        SAMPLE_RATE
    )

    sdr.gain = GAIN

    print(
        "[OK] RTL-SDR opened."
    )

    print(
        f"Sample rate: "
        f"{SAMPLE_RATE / 1e6:.3f} MS/s"
    )

    print(
        f"Gain: {GAIN} dB"
    )

    all_rows = []

    try:

        # ====================================================
        # STATION LOOP
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
                "=" * 70
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
                "=" * 70
            )

            # =================================================
            # TWO CAPTURES PER STATION
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
                # ONE RTL-SDR CAPTURE
                # ---------------------------------------------

                samples = capture_signal(
                    sdr,
                    frequency
                )

                # ---------------------------------------------
                # EXACT 1024 SAMPLES USED BY EVERY MODEL
                # ---------------------------------------------

                model_samples = (
                    samples[
                        :MODEL_INPUT_SAMPLES
                    ]
                    .copy()
                )

                # ---------------------------------------------
                # SAVE EXACT I/Q INPUT
                # ---------------------------------------------

                iq_filename = (
                    f"{station_name}_"
                    f"capture_{capture_number}.npy"
                )

                iq_path = (
                    IQ_DIR /
                    iq_filename
                )

                np.save(
                    iq_path,
                    model_samples
                )

                print(
                    f"[OK] I/Q saved: "
                    f"{iq_path}"
                )

                # ---------------------------------------------
                # CREATE ONE SHARED SPECTROGRAM
                # ---------------------------------------------

                (
                    image,
                    tensor
                ) = iq_to_spectrogram(
                    model_samples
                )

                spectrogram_path = (
                    save_spectrogram(
                        image,
                        station_name,
                        capture_number
                    )
                )

                print(
                    "[OK] One shared "
                    "spectrogram generated."
                )

                print(
                    f"[OK] Spectrogram saved: "
                    f"{spectrogram_path}"
                )

                # =============================================
                # SAME TENSOR -> ALL FIVE MODELS
                # =============================================

                for (
                    model_name,
                    model_data
                ) in loaded_models.items():

                    (
                        model,
                        output_classes
                    ) = model_data

                    (
                        top_classes,
                        top_probabilities
                    ) = predict(
                        model,
                        output_classes,
                        tensor,
                        device
                    )

                    fm_top1 = int(
                        top_classes[0]
                        ==
                        FM_CLASS_NAME
                    )

                    fm_top3 = int(
                        FM_CLASS_NAME
                        in
                        top_classes
                    )

                    row = {
                        "station":
                            station_name,

                        "frequency_mhz":
                            frequency / 1e6,

                        "capture":
                            capture_number,

                        "model":
                            model_name,

                        "top1":
                            top_classes[0],

                        "top1_probability":
                            float(
                                top_probabilities[0]
                            ),

                        "top2":
                            top_classes[1],

                        "top2_probability":
                            float(
                                top_probabilities[1]
                            ),

                        "top3":
                            top_classes[2],

                        "top3_probability":
                            float(
                                top_probabilities[2]
                            ),

                        "fm_top1_correct":
                            fm_top1,

                        "fm_in_top3":
                            fm_top3,

                        "spectrogram":
                            str(
                                spectrogram_path
                            ),

                        "iq_file":
                            str(
                                iq_path
                            ),
                    }

                    all_rows.append(
                        row
                    )

                    print(
                        f"\n  {model_name}"
                    )

                    print(
                        f"    Top-1: "
                        f"{top_classes[0]} "
                        f"({top_probabilities[0]:.2f}%)"
                    )

                    print(
                        f"    Top-2: "
                        f"{top_classes[1]} "
                        f"({top_probabilities[1]:.2f}%)"
                    )

                    print(
                        f"    Top-3: "
                        f"{top_classes[2]} "
                        f"({top_probabilities[2]:.2f}%)"
                    )

                    print(
                        f"    FM Top-1: "
                        f"{'YES' if fm_top1 else 'NO'}"
                    )

                    print(
                        f"    FM in Top-3: "
                        f"{'YES' if fm_top3 else 'NO'}"
                    )

                # ---------------------------------------------
                # SAVE PARTIAL RESULTS AFTER EVERY CAPTURE
                # ---------------------------------------------

                save_results_csv(
                    all_rows
                )

                del samples
                del model_samples
                del image
                del tensor

                gc.collect()

                if device.type == "cuda":

                    torch.cuda.empty_cache()

    finally:

        # ====================================================
        # ALWAYS RELEASE RTL-SDR
        # ====================================================

        print(
            "\nClosing RTL-SDR..."
        )

        sdr.close()

        print(
            "[OK] RTL-SDR closed."
        )

    # ========================================================
    # CHECK EXPECTED NUMBER OF RESULTS
    # ========================================================

    expected_captures = (
        len(STATIONS)
        *
        CAPTURES_PER_STATION
    )

    expected_rows = (
        expected_captures
        *
        len(MODEL_PATHS)
    )

    print(
        f"\nExpected captures: "
        f"{expected_captures}"
    )

    print(
        f"Expected prediction rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual prediction rows: "
        f"{len(all_rows)}"
    )

    if len(all_rows) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} "
            f"prediction rows, "
            f"but got {len(all_rows)}."
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    summary = calculate_summary(
        all_rows
    )

    save_results_csv(
        all_rows
    )

    save_summary_csv(
        summary
    )

    # ========================================================
    # GRAPHICAL MATERIALS FOR THESIS
    # ========================================================

    create_summary_table(
        summary
    )

    create_fm_rate_plot(
        summary
    )

    create_station_plot(
        all_rows
    )

    create_radioml_vs_real_plot(
        summary
    )

    save_text_report(
        summary
    )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "FINAL RTL-SDR FM RESULTS"
    )

    print(
        "=" * 70
    )

    for row in summary:

        print(
            f"\n{row['Model']}"
        )

        print(
            f"  FM Top-1: "
            f"{row['FM Top-1 Correct']}/"
            f"{row['Captures']} "
            f"= "
            f"{row['FM Top-1 Rate'] * 100:.2f}%"
        )

        print(
            f"  FM Top-3: "
            f"{row['FM Top-3 Correct']}/"
            f"{row['Captures']} "
            f"= "
            f"{row['FM Top-3 Rate'] * 100:.2f}%"
        )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "\nGenerated:"
    )

    print(
        f"  - {expected_captures} "
        f"real RTL-SDR captures"
    )

    print(
        f"  - {expected_captures} "
        f"shared spectrograms"
    )

    print(
        f"  - {expected_captures} "
        f"I/Q .npy files"
    )

    print(
        f"  - {expected_rows} "
        f"model predictions"
    )

    print(
        "  - FM Top-1 / Top-3 summary"
    )

    print(
        "  - 300 dpi summary table"
    )

    print(
        "  - 300 dpi Top-1 / Top-3 plot"
    )

    print(
        "  - 300 dpi per-station plot"
    )

    print(
        "  - 300 dpi RadioML vs RTL-SDR plot"
    )

    print(
        "\nDone."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()