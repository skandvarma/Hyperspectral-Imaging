import spectral.io.envi as envi
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings
import logging
import traceback
from validation_utils import validate_data_quality


def load_and_preprocess_data(hdr_path, output_dir=None):
    """
    Load and preprocess hyperspectral data with enhanced validation.

    Parameters:
    -----------
    hdr_path : str
        Path to either a .hdr file or directory containing .hdr files
    output_dir : str, optional
        Directory to save output visualization and processed data

    Returns:
    --------
    tuple
        (processed_data, wavelengths) or (processed_data_dict, wavelengths_dict)
    """
    try:
        if os.path.isdir(hdr_path):
            # Directory processing
            processed_data_dict = {}
            wavelengths_dict = {}

            # Get all HDR files
            hdr_files = [f for f in os.listdir(hdr_path) if f.endswith('.hdr')]
            if not hdr_files:
                raise FileNotFoundError(f"No .hdr files found in {hdr_path}")

            for hdr_file in hdr_files:
                try:
                    # Process each file
                    base_name = os.path.splitext(hdr_file)[0]
                    full_hdr_path = os.path.join(hdr_path, hdr_file)
                    dat_path = os.path.join(hdr_path, base_name + '.dat')

                    # Check for corresponding .dat file
                    if not os.path.exists(dat_path):
                        logging.warning(f"Could not find .dat file for {hdr_file}")
                        continue

                    # Load image data
                    img = envi.open(full_hdr_path, image=dat_path)
                    data = img.load()
                    wavelengths = np.array(img.bands.centers)

                    # Validate raw data
                    if data is None or wavelengths is None:
                        logging.warning(f"Invalid data or wavelengths for {hdr_file}")
                        continue

                    # Quality check on raw data
                    quality_metrics = validate_data_quality(data, wavelengths)
                    if not quality_metrics['valid_data']:
                        logging.warning(f"Quality issues in {hdr_file}: {quality_metrics['issues']}")
                        if len(quality_metrics['issues']) > 2:  # Skip if multiple serious issues
                            continue

                    # Preprocess data
                    processed_data = preprocess_spectral_data(data)
                    if processed_data is None:
                        logging.warning(f"Preprocessing failed for {hdr_file}")
                        continue

                    processed_data_dict[base_name] = processed_data
                    wavelengths_dict[base_name] = wavelengths

                    # Save preprocessed data and visualizations if output directory provided
                    if output_dir:
                        sample_output_dir = os.path.join(output_dir, base_name)
                        Path(sample_output_dir).mkdir(parents=True, exist_ok=True)

                        # Save preprocessed data
                        save_path = os.path.join(sample_output_dir, "preprocessed_data.npy")
                        np.save(save_path, processed_data)

                        # Create visualization
                        create_composite_visualization(processed_data, img, sample_output_dir)

                except Exception as e:
                    logging.error(f"Error processing file {hdr_file}: {str(e)}")
                    continue

            if not processed_data_dict:
                raise ValueError("No valid hyperspectral images were processed")

            return processed_data_dict, wavelengths_dict

        else:
            # Single file processing
            if not os.path.exists(hdr_path):
                raise FileNotFoundError(f"HDR file not found: {hdr_path}")

            base_name = os.path.splitext(os.path.basename(hdr_path))[0]
            dat_path = os.path.join(os.path.dirname(hdr_path), base_name + '.dat')

            if not os.path.exists(dat_path):
                raise FileNotFoundError(f"Could not find .dat file for {hdr_path}")

            # Load image data
            img = envi.open(hdr_path, image=dat_path)
            data = img.load()
            wavelengths = np.array(img.bands.centers)

            # Validate raw data
            quality_metrics = validate_data_quality(data, wavelengths)
            if not quality_metrics['valid_data']:
                logging.warning(f"Quality issues detected: {quality_metrics['issues']}")
                if len(quality_metrics['issues']) > 2:  # Multiple serious issues
                    raise ValueError(f"Data quality check failed: {quality_metrics['issues']}")

            # Preprocess data
            processed_data = preprocess_spectral_data(data)
            if processed_data is None:
                raise ValueError("Failed to preprocess data")

            # Save results if output directory provided
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                # Save preprocessed data
                save_path = os.path.join(output_dir, "preprocessed_data.npy")
                np.save(save_path, processed_data)

                # Create visualization
                create_composite_visualization(processed_data, img, output_dir)

            return processed_data, wavelengths

    except Exception as e:
        logging.error(f"Error processing hyperspectral image(s): {str(e)}")
        logging.error(traceback.format_exc())
        return None, None


# All other functions remain exactly the same
def preprocess_spectral_data(data):
    original_shape = data.shape
    processed_data = normalize_wavelengths(data)
    processed_data = remove_background_noise(processed_data)
    processed_data = align_spectral_bands(processed_data)
    return processed_data


def normalize_wavelengths(data, method='minmax'):
    rows, cols, bands = data.shape
    reshaped_data = data.reshape(-1, bands)

    if method == 'minmax':
        normalized = (reshaped_data - np.min(reshaped_data, axis=0)) / \
                     (np.max(reshaped_data, axis=0) - np.min(reshaped_data, axis=0))
    elif method == 'standard':
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped_data)

    return normalized.reshape(rows, cols, bands)


def remove_background_noise(data, threshold=0.1):
    mean_intensity = np.mean(data, axis=2)
    mask = mean_intensity > threshold
    cleaned_data = data.copy()
    cleaned_data[~mask] = 0
    return cleaned_data


def align_spectral_bands(data, window_size=3):
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    aligned_data = np.zeros_like(data)
    for band in range(data.shape[2]):
        aligned_data[:, :, band] = signal.convolve2d(
            data[:, :, band],
            kernel,
            mode='same',
            boundary='wrap'
        )
    return aligned_data


def create_composite_visualization(data, img, output_dir):
    rows, cols, bands = data.shape
    reshaped_data = data.reshape(-1, bands)

    pca = PCA(n_components=3)
    compressed_data = pca.fit_transform(reshaped_data)
    composite = compressed_data.reshape(rows, cols, 3)
    composite = (composite - np.min(composite)) / (np.max(composite) - np.min(composite))

    plt.figure(figsize=(12, 12))
    plt.imshow(composite)
    plt.axis('off')
    plt.title('Composite Image (Preprocessed Data)')

    save_path = os.path.join(output_dir, "composite_preprocessed.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    wavelengths = img.bands.centers
    print("\nSpectral Information:")
    print(f"Wavelength range: {min(wavelengths):.2f}nm - {max(wavelengths):.2f}nm")
    print(f"Number of bands combined: {len(wavelengths)}")
    print(f"\nVariance explained by PCA components:")
    print(f"R component: {pca.explained_variance_ratio_[0] * 100:.1f}%")
    print(f"G component: {pca.explained_variance_ratio_[1] * 100:.1f}%")
    print(f"B component: {pca.explained_variance_ratio_[2] * 100:.1f}%")


if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "hyperspectral_data")  # Directory containing .hdr files
    output_directory = os.path.join(current_dir, "preprocessed_data")

    processed_data, wavelengths = load_and_preprocess_data(data_dir, output_directory)

    if processed_data is not None and wavelengths is not None:
        if isinstance(processed_data, dict):
            print(f"\nProcessed {len(processed_data)} hyperspectral images")
            for name in processed_data.keys():
                print(f"Successfully processed: {name}")
        else:
            print("\nSuccessfully processed single hyperspectral image")