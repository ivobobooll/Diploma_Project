import io
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import (
    models,
    transforms
)

from PIL import Image

from rtlsdr import RtlSdr


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "resnet18_radioml_v2.pth"

NUM_CLASSES = 24

CENTER_FREQ = 104.6e6

SAMPLE_RATE = 2.048e6

GAIN = 40.0

# Capture more samples to flush / stabilize reception.
NUM_SAMPLES = 1024 * 64

# IMPORTANT:
# The neural network receives 1024 samples,
# matching the length of RadioML examples.
MODEL_INPUT_SAMPLES = 1024

DEBUG_IMAGE_PATH = (
    "debug_spectrogram_v2.png"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    print(
        "Loading ResNet18 v2 checkpoint..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # --------------------------------------------------------
    # Read class mapping saved during training
    # --------------------------------------------------------

    classes = checkpoint[
        "classes"
    ]

    class_to_idx = checkpoint[
        "class_to_idx"
    ]

    print(
        "\nClass mapping loaded "
        "from checkpoint:"
    )

    for class_name, index in (
        class_to_idx.items()
    ):

        print(
            f"{index:2d} -> "
            f"{class_name}"
        )

    if len(classes) != NUM_CLASSES:

        raise ValueError(
            f"Checkpoint contains "
            f"{len(classes)} classes, "
            f"expected {NUM_CLASSES}."
        )

    # --------------------------------------------------------
    # Initialize architecture
    # --------------------------------------------------------

    # No ImageNet weights are required here,
    # because trained weights will immediately be loaded.
    model = models.resnet18(
        weights=None
    )

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    model.eval()

    print(
        "\nModel loaded successfully."
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
    )

    print(
        f"Best training epoch: "
        f"{checkpoint['best_epoch']}"
    )

    return model, classes, checkpoint


# ============================================================
# CAPTURE RTL-SDR DATA
# ============================================================

def capture_sdr_data():

    print(
        "\nConnecting to RTL-SDR..."
    )

    print(
        f"Center frequency: "
        f"{CENTER_FREQ / 1e6:.6f} MHz"
    )

    print(
        f"Sample rate: "
        f"{SAMPLE_RATE / 1e6:.3f} MS/s"
    )

    print(
        f"Gain: {GAIN} dB"
    )

    sdr = RtlSdr()

    try:

        sdr.sample_rate = SAMPLE_RATE

        sdr.center_freq = CENTER_FREQ

        sdr.gain = GAIN

        print(
            "\nWarming up receiver..."
        )

        time.sleep(0.5)

        # ----------------------------------------------------
        # Flush buffers
        # ----------------------------------------------------

        print(
            "Flushing SDR buffers..."
        )

        for _ in range(10):

            _ = sdr.read_samples(
                1024 * 64
            )

            time.sleep(0.1)

        # ----------------------------------------------------
        # Capture
        # ----------------------------------------------------

        print(
            "Capturing live I/Q data..."
        )

        samples = sdr.read_samples(
            NUM_SAMPLES
        )

        print(
            f"Captured samples: "
            f"{len(samples)}"
        )

    finally:

        sdr.close()

    # --------------------------------------------------------
    # Match RadioML sample length
    # --------------------------------------------------------

    model_samples = samples[
        :MODEL_INPUT_SAMPLES
    ]

    print(
        f"Samples used by model: "
        f"{len(model_samples)}"
    )

    # Physical duration of the observation window
    duration = (
        len(model_samples)
        / SAMPLE_RATE
    )

    print(
        f"Observation duration: "
        f"{duration * 1000:.3f} ms"
    )

    return model_samples


# ============================================================
# CREATE SPECTROGRAM
# ============================================================

def iq_to_spectrogram_image(
    samples,
    checkpoint
):

    # --------------------------------------------------------
    # Read preprocessing parameters directly from checkpoint
    # --------------------------------------------------------

    nfft = checkpoint[
        "nfft"
    ]

    noverlap = checkpoint[
        "noverlap"
    ]

    spectrogram_fs = checkpoint[
        "spectrogram_fs"
    ]

    # --------------------------------------------------------
    # EXACTLY 224x224
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(2.24, 2.24),
        dpi=100
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    ax.axis("off")

    # --------------------------------------------------------
    # IMPORTANT:
    # Same spectrogram parameters as dataset generation
    # --------------------------------------------------------

    ax.specgram(
        samples,
        NFFT=nfft,
        Fs=spectrogram_fs,
        noverlap=noverlap,
        cmap="viridis"
    )

    # --------------------------------------------------------
    # Save debug image
    # --------------------------------------------------------

    fig.savefig(
        DEBUG_IMAGE_PATH,
        dpi=100,
        pad_inches=0
    )

    # --------------------------------------------------------
    # Save exactly the same figure to memory
    # --------------------------------------------------------

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=100,
        pad_inches=0
    )

    plt.close(fig)

    buffer.seek(0)

    image = Image.open(
        buffer
    ).convert("RGB")

    print(
        f"\nSpectrogram image size: "
        f"{image.size}"
    )

    if image.size != (
        224,
        224
    ):

        raise ValueError(
            "Spectrogram is not 224x224. "
            f"Actual size: {image.size}"
        )

    print(
        f"Debug spectrogram saved as: "
        f"{DEBUG_IMAGE_PATH}"
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "RTL-SDR + ResNet18 "
        "Modulation Classifier v2"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # 1. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # 2. Image transformation
    # --------------------------------------------------------

    transform = transforms.Compose([

        # Safety check.
        # Normally image is already 224x224.
        transforms.Resize(
            (224, 224)
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

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------

    try:

        model, classes, checkpoint = (
            load_model(device)
        )

    except FileNotFoundError:

        print(
            f"\nERROR: Model file "
            f"'{MODEL_PATH}' not found."
        )

        return

    except Exception as e:

        print(
            f"\nMODEL ERROR: {e}"
        )

        return

    # --------------------------------------------------------
    # 4. Capture SDR signal
    # --------------------------------------------------------

    try:

        samples = capture_sdr_data()

    except Exception as e:

        print(
            f"\nHARDWARE ERROR: {e}"
        )

        return

    # --------------------------------------------------------
    # 5. Spectrogram
    # --------------------------------------------------------

    try:

        image = (
            iq_to_spectrogram_image(
                samples,
                checkpoint
            )
        )

    except Exception as e:

        print(
            f"\nSPECTROGRAM ERROR: {e}"
        )

        return

    # --------------------------------------------------------
    # 6. Prepare tensor
    # --------------------------------------------------------

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Network input tensor: "
        f"{tuple(input_tensor.shape)}"
    )

    # Expected:
    # (1, 3, 224, 224)

    # --------------------------------------------------------
    # 7. Inference
    # --------------------------------------------------------

    print(
        "\nAnalyzing signal..."
    )

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = (
            torch.nn.functional.softmax(
                outputs,
                dim=1
            )
        )

        top3_probabilities, top3_indices = (
            torch.topk(
                probabilities,
                k=3,
                dim=1
            )
        )

    # --------------------------------------------------------
    # 8. Results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CLASSIFICATION RESULTS"
    )

    print(
        "=" * 60
    )

    for rank in range(3):

        probability = (
            top3_probabilities[
                0
            ][rank].item()
            * 100
        )

        class_index = (
            top3_indices[
                0
            ][rank].item()
        )

        class_name = classes[
            class_index
        ]

        print(
            f"{rank + 1}. "
            f"{class_name:12s} "
            f"{probability:6.2f}%"
        )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()