import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

# --- CONFIGURATION ---
HDF5_FILE_PATH = "/home/lily/Downloads/DIPLOMA/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
DATASET_DIR = "dataset_images/train"
MIN_SNR = 10  # Curriculum Learning: start with clean signals (>= 10 dB)
SAMPLES_PER_CLASS = 5000  # Number of spectrograms to generate per class for the initial test

# 24 modulation classes (order matches DeepSig RadioML 2018.01A format)
CLASSES = [
    "OOK", "ASK4", "ASK8", "BPSK", "QPSK", "PSK8", "PSK16", "PSK32",
    "APSK16", "APSK32", "APSK64", "APSK128", "QAM16", "QAM32", "QAM64",
    "QAM128", "QAM256", "AM_SSB_WC", "AM_SSB_SC", "AM_DSB_WC",
    "AM_DSB_SC", "FM", "GMSK", "OQPSK"
]


def create_folders():
    """Create directory structure for each modulation class."""
    for class_name in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)


def draw_and_save_spectrogram(iq_data, save_path):
    """Generate and save a spectrogram image from I/Q data."""
    # iq_data shape is (1024, 2) - combine I and Q into a complex signal
    complex_signal = iq_data[:, 0] + 1j * iq_data[:, 1]

    # Set target image size to 224x224 pixels (standard input size for ResNet18)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(complex_signal, NFFT=64, Fs=1000, noverlap=32, cmap='viridis')
    plt.axis('off')  # Remove axes for clean image processing
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    print("1. Creating directories...")
    create_folders()

    print(f"2. Opening dataset file: {HDF5_FILE_PATH}...")
    try:
        with h5py.File(HDF5_FILE_PATH, 'r') as f:
            # Print available keys to verify HDF5 internal structure
            print(f"Available HDF5 keys: {list(f.keys())}")

            X = f['X']  # Signal data
            Y = f['Y']  # One-hot encoded labels
            Z = f['Z']  # SNR values

            total_samples = X.shape[0]
            print(f"Total samples in dataset: {total_samples}")

            # Dictionary to track the number of saved images per class
            class_counters = {class_name: 0 for class_name in CLASSES}

            print("3. Searching for high-SNR signals and generating spectrograms...")
            for i in range(total_samples):
                snr = Z[i][0]

                # Filter signals based on defined Minimum SNR threshold
                if snr >= MIN_SNR:
                    class_index = np.argmax(Y[i])
                    class_name = CLASSES[class_index]

                    # Process only if the target sample limit for this class is not yet reached
                    if class_counters[class_name] < SAMPLES_PER_CLASS:
                        save_path = os.path.join(
                            DATASET_DIR,
                            class_name,
                            f"{class_name}_{snr}dB_{class_counters[class_name]}.png"
                        )

                        draw_and_save_spectrogram(X[i], save_path)
                        class_counters[class_name] += 1

                        # Print progress every 50 saved images
                        total_saved = sum(class_counters.values())
                        if total_saved % 50 == 0:
                            print(f"Saved images: {total_saved} / {SAMPLES_PER_CLASS * len(CLASSES)}")

                # Stop processing if all classes have reached the required sample limit
                if all(count >= SAMPLES_PER_CLASS for count in class_counters.values()):
                    print("Initial training set generation completed successfully!")
                    break

    except FileNotFoundError:
        print(f"\nERROR: File not found at path {HDF5_FILE_PATH}")
        print("Ensure the absolute path is correct.")
    except KeyError as e:
        print(f"\nKEY ERROR: Array {e} is missing in the file. Check 'Available HDF5 keys' above.")


if __name__ == "__main__":
    main()