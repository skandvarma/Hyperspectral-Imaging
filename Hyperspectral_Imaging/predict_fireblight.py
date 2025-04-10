from fireblight_detection import FireBlightDetectionPipeline
from data_processing_function import load_and_preprocess_data
from feature_extraction import (
    calculate_disease_indices,
    analyze_disease_patterns,
    calculate_vegetation_indices
)
from validation_utils import (
    setup_logger,
    validate_data_quality,
    cross_validate_indicators,
    filter_false_positives
)
from matplotlib import patches
from skimage import measure, morphology
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import traceback

def numpy_to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_disease_visualization(processed_data, wavelengths, prediction, output_path):
    """
    Create visualization of disease detection results with refined lesion detection
    focusing on the red areas in NDVI that indicate potential disease.
    """
    try:
        # Calculate disease indices for visualization
        disease_indices = calculate_disease_indices(processed_data, wavelengths)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3)

        # Initialize subplot axes
        ax1 = fig.add_subplot(gs[0, 0])  # Disease NDVI
        ax2 = fig.add_subplot(gs[0, 1])  # Green-NIR Ratio
        ax3 = fig.add_subplot(gs[0, 2])  # Disease Severity Map
        ax4 = fig.add_subplot(gs[1, 0])  # RGB with Lesion Detection
        ax5 = fig.add_subplot(gs[1, 1:])  # Prediction Results

        try:
            # Create refined leaf mask using NDVI
            ndvi = disease_indices['disease_NDVI']
            leaf_mask = (ndvi > -0.15) & (ndvi < 0.8)

            # Refined lesion detection focusing on high NDVI values (reddish areas in visualization)
            lesion_mask = (ndvi > 0.35) & (ndvi < 0.60060) & leaf_mask

            # Enhanced morphological operations for better lesion definition
            lesion_mask = morphology.remove_small_objects(lesion_mask, min_size=100)
            lesion_mask = morphology.remove_small_holes(lesion_mask, area_threshold=20)
            lesion_mask = morphology.binary_erosion(lesion_mask, morphology.disk(2))
            lesion_mask = morphology.binary_dilation(lesion_mask, morphology.disk(3))

            # Label the refined regions
            labeled_regions = measure.label(lesion_mask)

            # Get disease patterns with adjusted parameters
            disease_patterns, _, _ = analyze_disease_patterns(
                processed_data,
                wavelengths,
                min_lesion_size=100,
                max_lesion_size=2000
            )

        except Exception as e:
            logging.error(f"Error in disease pattern analysis: {str(e)}")
            disease_mask = np.zeros_like(disease_indices['disease_NDVI'], dtype=bool)
            labeled_regions = np.zeros_like(disease_indices['disease_NDVI'], dtype=int)

        # Plot 1: Disease NDVI with lesion bounding boxes
        im1 = ax1.imshow(disease_indices['disease_NDVI'], cmap='RdYlBu_r')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('Disease NDVI')

        # Plot lesion bounding boxes with refined criteria
        try:
            props = measure.regionprops(labeled_regions)
            valid_lesions = []

            for prop in props:
                # Refined lesion validation criteria
                if (30 < prop.area < 2000 and  # Adjusted maximum area
                        prop.eccentricity < 0.95 and  # More permissive eccentricity
                        prop.solidity > 0.3 and  # More permissive solidity
                        prop.extent > 0.2):  # More permissive extent

                    # Verify lesion is within leaf and has high NDVI
                    lesion_coords = prop.coords
                    if np.all(leaf_mask[lesion_coords[:, 0], lesion_coords[:, 1]]):
                        # Check average NDVI value in the region
                        region_ndvi = np.mean(ndvi[lesion_coords[:, 0], lesion_coords[:, 1]])
                        if region_ndvi > 0.2:  # Only include regions with high NDVI
                            valid_lesions.append(prop)

                            minr, minc, maxr, maxc = prop.bbox
                            rect = patches.Rectangle(
                                (minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2
                            )
                            ax1.add_patch(rect)

            prediction['features']['lesion_analysis']['total_lesions'] = len(valid_lesions)
            if valid_lesions:
                prediction['features']['lesion_analysis']['avg_lesion_size'] = np.mean([p.area for p in valid_lesions])

        except Exception as e:
            logging.warning(f"Could not add bounding boxes to NDVI plot: {str(e)}")

        # Plot 2: Green-NIR Ratio
        im2 = ax2.imshow(disease_indices['green_NIR_ratio'], cmap='RdYlBu_r')
        ax2.set_title('Green-NIR Ratio')
        plt.colorbar(im2, ax=ax2)

        # Plot 3: Enhanced Disease Severity Map
        severity_map = create_severity_map(disease_indices, ndvi, leaf_mask)
        im3 = ax3.imshow(severity_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax3.set_title('Disease Severity Map')
        plt.colorbar(im3, ax=ax3)

        # Plot 4: RGB with Lesion Detection
        rgb_bands = create_rgb_composite(processed_data, wavelengths)
        ax4.imshow(rgb_bands)
        ax4.set_title('RGB with Lesion Detection')

        # Add lesion bounding boxes to RGB image
        for prop in valid_lesions:
            minr, minc, maxr, maxc = prop.bbox
            rect = patches.Rectangle(
                (minc, minr), maxc - minc, maxr - minr,
                fill=False, edgecolor='yellow', linewidth=2
            )
            ax4.add_patch(rect)

        # Plot 5: Prediction Results
        display_prediction_results(ax5, prediction)

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return prediction

    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        logging.error(f"Prediction structure: {json.dumps(numpy_to_python_types(prediction), indent=2)}")
        raise


def create_severity_map(disease_indices, ndvi, leaf_mask):
    """Create an enhanced severity map focusing on high NDVI areas."""
    green_nir = disease_indices['green_NIR_ratio']

    # Normalize NDVI to 0-1 range
    ndvi_norm = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-8)
    # Normalize Green-NIR ratio
    gnir_norm = (green_nir - np.min(green_nir)) / (np.max(green_nir) - np.min(green_nir) + 1e-8)

    # Modified severity map calculation focusing on high NDVI areas
    severity_map = 0.7 * ndvi_norm + 0.3 * gnir_norm  # Increased weight on NDVI
    severity_threshold = 0.4  # Adjusted threshold

    # Apply leaf mask to severity map
    severity_map[~leaf_mask] = 0
    severity_map[severity_map < severity_threshold] = 0

    return severity_map


def create_rgb_composite(processed_data, wavelengths):
    """Create an RGB composite image from hyperspectral data."""
    rgb_bands = np.dstack([
        processed_data[:, :, np.argmin(np.abs(wavelengths - 670))],  # Red
        processed_data[:, :, np.argmin(np.abs(wavelengths - 550))],  # Green
        processed_data[:, :, np.argmin(np.abs(wavelengths - 450))]  # Blue
    ])
    return np.clip((rgb_bands - np.min(rgb_bands)) / (np.max(rgb_bands) - np.min(rgb_bands) + 1e-8), 0, 1)


def display_prediction_results(ax, prediction):
    """Display prediction results with formatted text."""
    ax.axis('off')

    result_text = (
        f"Prediction Results:\n"
        f"Disease Stage: {prediction['predicted_stage']}\n"
        f"Confidence: {prediction['confidence']:.2f}\n\n"
        f"Disease Features:\n"
        # f"• Disease Severity: {prediction['features']['affected_area']['severity']:.3f}\n"
        # f"• Affected Area Ratio: {prediction['features']['affected_area']['ratio']:.3f}\n"
        f"• Lesion Count: {prediction['features']['lesion_analysis']['total_lesions']}\n"
        f"• Pattern Uniformity: {prediction['features']['lesion_analysis']['pattern_uniformity']:.3f}\n"
        f"• Average Lesion Size: {prediction['features']['lesion_analysis']['avg_lesion_size']:.1f}\n"
    )

    stage_colors = {
        'Healthy': 'green',
        'Early Stage': 'yellow',
        'Moderate': 'orange',
        'Severe': 'red',
        'Critical': 'darkred'
    }

    bbox_props = dict(
        boxstyle='round,pad=0.5',
        facecolor=stage_colors.get(prediction['predicted_stage'], 'gray'),
        alpha=0.3
    )

    ax.text(0.5, 0.5, result_text,
            fontsize=12,
            ha='center',
            va='center',
            bbox=bbox_props,
            transform=ax.transAxes)

def create_timestamped_dir(base_dir):
    """
    Create a timestamped directory for outputs.

    Parameters:
    -----------
    base_dir : str
        Base directory path

    Returns:
    --------
    str
        Path to timestamped directory
    """
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory structure
    timestamped_dir = os.path.join(base_dir, f"prediction_results_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)

    # Create subdirectories for different types of outputs
    subdirs = ['predictions', 'visualizations', 'summary']
    for subdir in subdirs:
        os.makedirs(os.path.join(timestamped_dir, subdir), exist_ok=True)

    return timestamped_dir


def predict_fireblight(hdr_path, model_path, output_dir=None):
    """Predict fire blight disease from hyperspectral image."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        # Load and preprocess the hyperspectral image
        logger.info(f"Loading and preprocessing image from: {hdr_path}")
        processed_data, wavelengths = load_and_preprocess_data(hdr_path)

        if processed_data is None or wavelengths is None:
            raise ValueError("Failed to load or preprocess the image")

        # Initialize the model with trained weights
        logger.info("Loading trained model")
        pipeline = FireBlightDetectionPipeline(model_path=model_path)

        # Get the day number from filename
        try:
            filename = os.path.basename(hdr_path)
            day = int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            logger.warning("Could not extract day number from filename, using 0")
            day = 0

        # Make prediction
        logger.info("Making prediction")
        prediction = pipeline.predict_quality(hdr_path, day)

        if output_dir:
            base_name = os.path.splitext(os.path.basename(hdr_path))[0]

            # Save prediction results as JSON with numpy handling
            json_path = os.path.join(output_dir, 'predictions', f"{base_name}_prediction.json")

            # Convert numpy arrays to lists before JSON serialization
            def numpy_to_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (list, tuple)):
                    return [numpy_to_json(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: numpy_to_json(value) for key, value in obj.items()}
                return obj

            serializable_prediction = numpy_to_json(prediction)

            with open(json_path, 'w') as f:
                json.dump(serializable_prediction, f, indent=4)
            logger.info(f"Results saved to: {json_path}")

            # Create and save visualization
            viz_path = os.path.join(output_dir, 'visualizations', f"{base_name}_visualization.png")
            create_disease_visualization(processed_data, wavelengths, prediction, viz_path)
            logger.info(f"Visualization saved to: {viz_path}")

        return prediction, processed_data, wavelengths

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def create_summary_report(results_dict, output_dir):
    """Create a summary report of all processed images."""
    summary_dir = os.path.join(output_dir, 'summary')

    # Prepare data for plotting
    stages = []
    confidences = []
    severities = []
    filenames = []

    # Collect data for summary
    for filename, (pred, _, _) in results_dict.items():
        stages.append(pred['predicted_stage'])
        confidences.append(float(pred['confidence']))  # Convert numpy float to Python float

        # Handle different data structures for severity
        if 'affected_area' in pred['features']:
            severity = float(pred['features']['affected_area']['severity'])  # Convert to Python float
        elif 'disease_patterns' in pred['features']:
            severity = float(pred['features']['disease_patterns']['disease_severity'])  # Convert to Python float
        else:
            severity = 0.0
            print(f"Warning: Could not find severity value for {filename}")

        severities.append(severity)
        filenames.append(filename)

    # Create summary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Disease stages distribution
    stage_counts = sns.countplot(x=stages, ax=ax1)
    ax1.set_title('Distribution of Disease Stages')

    # Get the current tick positions
    ticks = ax1.get_xticks()
    # Get the current tick labels
    labels = [item.get_text() for item in ax1.get_xticklabels()]

    # Set ticks and labels explicitly
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Plot 2: Confidence vs Severity
    ax2.scatter(confidences, severities)
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Disease Severity')
    ax2.set_title('Confidence vs Disease Severity')

    # Add labels for points
    for i, filename in enumerate(filenames):
        ax2.annotate(os.path.basename(filename),
                     (confidences[i], severities[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'summary_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed results as CSV with explicit type conversion
    import pandas as pd
    summary_df = pd.DataFrame({
        'Filename': filenames,
        'Disease_Stage': stages,
        'Confidence': [float(x) for x in confidences],  # Convert all numpy values to Python types
        'Severity': [float(x) for x in severities]
    })
    summary_df.to_csv(os.path.join(summary_dir, 'detailed_results.csv'), index=False)

    return summary_df


def main():
    """Example usage of the prediction script with support for both directory and single file"""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict fire blight from hyperspectral images')
    parser.add_argument('input_path', type=str, help='Path to .hdr file or directory containing .hdr files')
    parser.add_argument('--model_path', type=str, default="results/best_model.keras", help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default="prediction_results", help='Base directory for results')

    args = parser.parse_args()

    # Create timestamped output directory
    output_dir = create_timestamped_dir(args.output_dir)
    print(f"Results will be saved in: {output_dir}")

    # Set up logging file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'processing.log')),
            logging.StreamHandler()
        ]
    )

    results_dict = {}

    try:
        if os.path.isfile(args.input_path):
            # Process single .hdr file
            if not args.input_path.endswith('.hdr'):
                raise ValueError("Single file input must be a .hdr file")

            print(f"\nProcessing single file: {args.input_path}")
            results = predict_fireblight(args.input_path, args.model_path, output_dir)
            results_dict[os.path.basename(args.input_path)] = results

            # Print prediction results
            prediction = results[0]
            print("\nPrediction Results:")
            print(f"Disease Stage: {prediction['predicted_stage']}")
            print(f"Confidence: {prediction['confidence']:.2f}")
            print("\nDisease Features:")
            for feature, value in prediction['features'].items():
                print(f"{feature}: {value}")

        elif os.path.isdir(args.input_path):
            # Process directory of .hdr files
            hdr_files = [f for f in os.listdir(args.input_path) if f.endswith('.hdr')]

            if not hdr_files:
                raise ValueError(f"No .hdr files found in directory: {args.input_path}")

            for hdr_file in hdr_files:
                hdr_path = os.path.join(args.input_path, hdr_file)
                print(f"\nProcessing: {hdr_file}")

                try:
                    # Get predictions and data
                    results = predict_fireblight(hdr_path, args.model_path, output_dir)
                    results_dict[hdr_file] = results

                    # Print prediction results
                    prediction = results[0]
                    print("\nPrediction Results:")
                    print(f"Disease Stage: {prediction['predicted_stage']}")
                    print(f"Confidence: {prediction['confidence']:.2f}")
                    print("\nDisease Features:")
                    for feature, value in prediction['features'].items():
                        print(f"{feature}: {value}")

                except Exception as e:
                    print(f"Error processing {hdr_file}: {str(e)}")
                    logging.error(traceback.format_exc())
                    continue
        else:
            raise ValueError(f"Input path does not exist: {args.input_path}")

        # Create summary report if we have results
        if results_dict:
            create_summary_report(results_dict, output_dir)
            print(f"\nProcessing complete. Results saved in: {output_dir}")
            print(f"Summary report available at: {os.path.join(output_dir, 'summary')}")
        else:
            print("\nNo results were generated.")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()