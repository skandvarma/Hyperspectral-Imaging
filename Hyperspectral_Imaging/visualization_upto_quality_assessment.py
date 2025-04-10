import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
from quality_assessment_function import assess_overall_quality
from feature_extraction import extract_all_features
from data_processing_function import load_and_preprocess_data
import os
from datetime import datetime


def create_quality_dashboard(hdr_file):
    """
    Create a comprehensive dashboard visualization of leaf quality metrics.
    """
    try:
        # Load and process data
        print("Loading and processing data...")
        data, wavelengths = load_and_preprocess_data(hdr_file)
        if data is None or wavelengths is None:
            print("Error: Could not load data")
            return None

        # Convert wavelengths to numpy array if it's a list
        wavelengths = np.array(wavelengths, dtype=float)

        # Get quality assessment and features
        print("Performing quality assessment...")
        quality_results = assess_overall_quality(data, wavelengths)
        features = extract_all_features(data, wavelengths)

        # Create dashboard
        fig = plt.figure(figsize=(25, 22))  # Increased height to accommodate metrics
        gs = GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 0.5])
        fig.suptitle('Leaf Analysis Dashboard', fontsize=16, y=0.95)

        # 1. Chlorophyll and Water Content Plot
        print("Creating health indicators plot...")
        ax1 = fig.add_subplot(gs[0, 0])
        if quality_results['chlorophyll'] and quality_results['water']:
            chloro_score = quality_results['chlorophyll']['average_chlorophyll']
            water_score = quality_results['water']['average_water_content']

            data_health = pd.DataFrame({
                'Metrics': ['Chlorophyll', 'Water Content'],
                'Values': [chloro_score, water_score]
            })

            sns.barplot(data=data_health, x='Metrics', y='Values', ax=ax1)
            ax1.set_title('Leaf Health Indicators')
            ax1.set_ylim([0, max(chloro_score, water_score) * 1.2])

        # 2. Vegetation Indices Distribution
        print("Creating vegetation indices plot...")
        ax2 = fig.add_subplot(gs[0, 1])
        if features and 'vegetation_indices' in features:
            indices_data = pd.DataFrame({
                'NDVI': features['vegetation_indices']['NDVI'].flatten(),
                'GNDVI': features['vegetation_indices']['GNDVI'].flatten(),
                'NDRE': features['vegetation_indices']['NDRE'].flatten()
            })
            indices_data.boxplot(ax=ax2)
            ax2.set_title('Vegetation Indices Distribution')
            ax2.set_ylabel('Index Value')
            ax2.tick_params(axis='x', rotation=45)

        # 3. Leaf Health Map (NDVI)
        print("Creating leaf health map...")
        ax3 = fig.add_subplot(gs[0, 2])
        if features and 'vegetation_indices' in features:
            ndvi = features['vegetation_indices']['NDVI']
            im = ax3.imshow(ndvi, cmap='RdYlGn')
            plt.colorbar(im, ax=ax3)
            ax3.set_title('Leaf Health Map (NDVI)')
            ax3.axis('off')

        # 4. Texture Analysis
        print("Creating texture analysis plot...")
        ax4 = fig.add_subplot(gs[1, 0])
        if 'structure' in quality_results:
            texture_metrics = pd.DataFrame({
                'Metric': ['Contrast', 'Homogeneity', 'Energy', 'ASM'],
                'Value': [
                    quality_results['structure']['texture_contrast'],
                    quality_results['structure']['texture_homogeneity'],
                    quality_results['structure']['surface_uniformity'],
                    features['texture_features']['ASM']
                ]
            })
            sns.barplot(data=texture_metrics, x='Metric', y='Value', ax=ax4)
            ax4.set_title('Leaf Texture Analysis')
            ax4.tick_params(axis='x', rotation=45)

        # 5. Stress Indicators
        print("Creating stress indicators plot...")
        ax5 = fig.add_subplot(gs[1, 1])
        if 'stress' in quality_results:
            stress_metrics = pd.DataFrame({
                'Metric': ['Tissue Uniformity', 'Stress Indicator', 'Tissue Integrity'],
                'Value': [
                    quality_results['stress']['tissue_uniformity'],
                    quality_results['stress']['stress_indicator'],
                    quality_results['stress']['tissue_integrity']
                ]
            })
            sns.barplot(data=stress_metrics, x='Metric', y='Value', ax=ax5)
            ax5.set_title('Stress Indicators')
            ax5.tick_params(axis='x', rotation=45)

        # 6. Quality Grades Summary
        print("Creating quality grades summary...")
        ax6 = fig.add_subplot(gs[1, 2])
        quality_grades = pd.DataFrame({
            'Category': ['Chlorophyll', 'Hydration', 'Structure', 'Stress', 'Damage'],
            'Grade': [
                quality_results['chlorophyll']['quality_grade'],
                quality_results['water']['hydration_status'],
                quality_results['structure']['structure_grade'],
                quality_results['stress']['stress_level'],
                quality_results['damage']['leaf_condition']
            ]
        })

        grade_map = {
            'Healthy': 3, 'Well-hydrated': 3, 'Low': 3,
            'Moderate': 2, 'Moderately hydrated': 2, 'Minor damage': 2,
            'Poor': 1, 'Water stressed': 1, 'High': 1, 'Significant damage': 1
        }

        quality_grades['Numeric'] = quality_grades['Grade'].map(grade_map)

        sns.barplot(data=quality_grades, x='Category', y='Numeric', ax=ax6)
        ax6.set_title('Leaf Quality Assessment')
        ax6.set_ylim([0, 3.5])
        ax6.tick_params(axis='x', rotation=45)

        # 7. Spectral Profile
        print("Creating spectral profile plot...")
        ax7 = fig.add_subplot(gs[2, :])
        mean_spectrum = np.mean(data, axis=(0, 1))
        ax7.plot(wavelengths, mean_spectrum, label='Mean Spectrum')

        # Find and highlight red edge region
        red_edge_start_idx = np.argmin(np.abs(wavelengths - 670))
        red_edge_end_idx = np.argmin(np.abs(wavelengths - 780))

        if red_edge_start_idx < red_edge_end_idx:
            ax7.axvspan(wavelengths[red_edge_start_idx], wavelengths[red_edge_end_idx],
                        alpha=0.2, color='red', label='Red Edge Region')

        ax7.set_title('Leaf Spectral Profile')
        ax7.set_xlabel('Wavelength (nm)')
        ax7.set_ylabel('Reflectance')
        ax7.grid(True)
        ax7.legend()

        # 8. Disease Detection Map
        print("Creating disease detection visualization...")
        ax8 = fig.add_subplot(gs[3, 0])
        if 'disease_indices' in features:
            disease_ndvi = features['disease_indices']['disease_NDVI']
            im = ax8.imshow(disease_ndvi, cmap='RdYlBu_r')
            plt.colorbar(im, ax=ax8)
            ax8.set_title('Disease Detection Map')
            ax8.axis('off')

        # 9. Disease Severity Analysis
        print("Creating disease severity plot...")
        ax9 = fig.add_subplot(gs[3, 1])
        if 'disease' in quality_results:
            severity_metrics = pd.DataFrame({
                'Metric': ['Affected Area', 'Disease Severity', 'Risk Level'],
                'Value': [
                    quality_results['disease']['affected_area_ratio'],
                    quality_results['disease']['disease_severity'],
                    quality_results['disease']['risk_level']
                ]
            })
            sns.barplot(data=severity_metrics, x='Metric', y='Value', ax=ax9)
            ax9.set_title('Disease Severity Metrics')
            ax9.tick_params(axis='x', rotation=45)

        # 10. Disease Pattern Analysis
        print("Creating disease pattern analysis...")
        ax10 = fig.add_subplot(gs[3, 2])
        if 'disease' in quality_results and 'lesion_count' in quality_results['disease']:
            pattern_metrics = pd.DataFrame({
                'Metric': ['Lesion Count', 'Avg Lesion Size', 'Pattern Uniformity'],
                'Value': [
                    quality_results['disease']['lesion_count'],
                    quality_results['disease']['avg_lesion_size'],
                    quality_results['disease']['pattern_uniformity']
                ]
            })
            sns.barplot(data=pattern_metrics, x='Metric', y='Value', ax=ax10)
            ax10.set_title('Disease Pattern Analysis')
            ax10.tick_params(axis='x', rotation=45)

        # Add metrics text in the bottom row
        print("Adding detailed metrics...")
        ax11 = fig.add_subplot(gs[4, :])
        ax11.axis('off')

        metrics_text = (
            f"DETAILED METRICS:\n\n"
            f"CHLOROPHYLL:\n"
            f"• Average: {quality_results['chlorophyll']['average_chlorophyll']:.4f}\n"
            f"• Uniformity: {quality_results['chlorophyll']['chlorophyll_uniformity']:.4f}\n"
            f"• NDVI Correlation: {quality_results['chlorophyll']['ndvi_correlation']:.4f}\n"
            f"• Quality Grade: {quality_results['chlorophyll']['quality_grade']}\n\n"

            f"WATER CONTENT:\n"
            f"• Average: {quality_results['water']['average_water_content']:.4f}\n"
            f"• Distribution: {quality_results['water']['water_distribution']:.4f}\n"
            f"• Gradient: {quality_results['water']['water_gradient']:.6f}\n"
            f"• Status: {quality_results['water']['hydration_status']}\n\n"

            f"DISEASE STATUS:\n"
            f"• Status: {quality_results['disease']['disease_status']}\n"
            f"• Confidence: {quality_results['disease']['confidence']}\n"
            f"• Affected Area: {quality_results['disease']['affected_area_ratio']:.4f}\n"
            f"• Severity: {quality_results['disease']['disease_severity']:.4f}\n"
            f"• Risk Level: {quality_results['disease']['risk_level']:.4f}\n\n"

            f"STRUCTURE & STRESS:\n"
            f"• Texture Contrast: {quality_results['structure']['texture_contrast']:.4f}\n"
            f"• Surface Uniformity: {quality_results['structure']['surface_uniformity']:.4f}\n"
            f"• Structure Grade: {quality_results['structure']['structure_grade']}\n"
            f"• Stress Level: {quality_results['stress']['stress_level']}"
        )

        # Add text box with metrics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        ax11.text(0.01, 0.99, metrics_text, transform=ax11.transAxes,
                  fontsize=10, verticalalignment='top',
                  family='monospace', bbox=props)

        # Add metadata box
        metadata_text = (
            f"Image Statistics:\n"
            f"• Wavelength Range: {wavelengths[0]:.1f}nm - {wavelengths[-1]:.1f}nm\n"
            f"• Spatial Dimensions: {data.shape[0]}x{data.shape[1]}\n"
            f"• Spectral Bands: {data.shape[2]}\n"
            f"• Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"• Disease Analysis: Fire Blight Detection"
        )
        ax11.text(0.67, 0.99, metadata_text, transform=ax11.transAxes,
                  fontsize=10, verticalalignment='top',
                  family='monospace', bbox=props)

        # Save additional disease-specific results to text file
        results_path = os.path.join(os.path.dirname(hdr_file),
                                    f'disease_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        with open(results_path, 'w') as f:
            f.write("Disease Analysis Results\n")
            f.write("=======================\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Disease Type: Fire Blight\n\n")
            f.write("Disease Metrics:\n")
            for metric, value in quality_results['disease'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.6f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("\nRecommendations:\n")
            if quality_results['disease']['risk_level'] > 0.6:
                f.write("- Immediate treatment recommended\n")
                f.write("- Isolate affected areas\n")
                f.write("- Consider fungicide application\n")
            elif quality_results['disease']['risk_level'] > 0.3:
                f.write("- Regular monitoring required\n")
                f.write("- Preventive measures recommended\n")
            else:
                f.write("- Continue routine monitoring\n")

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_visualization_batch(data_dir, output_dir='visualization_output'):
    """
    Create and save visualizations and analysis results for all hyperspectral images in a directory.

    Parameters:
    -----------
    data_dir : str
        Directory containing hyperspectral data files (.hdr and .dat pairs)
    output_dir : str
        Directory to save the visualizations and results
    """
    try:
        # Create timestamp for this batch processing
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(output_dir, timestamp)
        os.makedirs(batch_dir, exist_ok=True)

        # Get all .hdr files in the directory
        hdr_files = [f for f in os.listdir(data_dir) if f.endswith('.hdr')]

        if not hdr_files:
            print("No .hdr files found in the directory")
            return

        # Process each file
        for hdr_file in hdr_files:
            try:
                full_hdr_path = os.path.join(data_dir, hdr_file)
                base_name = os.path.splitext(hdr_file)[0]

                # Create subdirectory for this image
                image_dir = os.path.join(batch_dir, base_name)
                os.makedirs(image_dir, exist_ok=True)

                print(f"\nProcessing file: {hdr_file}")
                fig = create_quality_dashboard(full_hdr_path)

                if fig is not None:
                    # Save dashboard
                    dashboard_path = os.path.join(image_dir, f'{base_name}_dashboard.png')
                    results_path = os.path.join(image_dir, f'{base_name}_analysis.txt')

                    fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Dashboard saved to: {dashboard_path}")

                    # Get quality results for text file
                    data, wavelengths = load_and_preprocess_data(full_hdr_path)
                    quality_results = assess_overall_quality(data, wavelengths)

                    # Save detailed results
                    with open(results_path, 'w') as f:
                        f.write(f"Quality Assessment Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=====================================\n\n")
                        f.write(f"Input File: {hdr_file}\n\n")

                        for category, results in quality_results.items():
                            if results:  # Check if results exist
                                f.write(f"{category.upper()}:\n")
                                for metric, value in results.items():
                                    if isinstance(value, (int, float)):
                                        f.write(f"  {metric}: {value:.6f}\n")
                                    elif isinstance(value, np.ndarray):
                                        f.write(f"  {metric}:\n")
                                        f.write(f"    Shape: {value.shape}\n")
                                        f.write(f"    Mean: {np.mean(value):.6f}\n")
                                        f.write(f"    Std: {np.std(value):.6f}\n")
                                        f.write(f"    Min: {np.min(value):.6f}\n")
                                        f.write(f"    Max: {np.max(value):.6f}\n")
                                    else:
                                        f.write(f"  {metric}: {value}\n")
                                f.write("\n")

                    print(f"Analysis results saved to: {results_path}")

            except Exception as e:
                print(f"Error processing {hdr_file}: {str(e)}")
                continue

        # Create a summary report for all processed images
        summary_path = os.path.join(batch_dir, 'batch_processing_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Batch Processing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=====================================\n\n")
            f.write(f"Total files processed: {len(hdr_files)}\n")
            f.write(f"Output directory: {batch_dir}\n\n")
            f.write("Processed files:\n")
            for hdr_file in hdr_files:
                f.write(f"- {hdr_file}\n")

        print(f"\nBatch processing complete. All results saved in: {batch_dir}")
        print(f"Summary report saved to: {summary_path}")

    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, "hyperspectral_data")
        output_dir = os.path.join(current_dir, "visualization_output")

        if not os.path.exists(data_dir):
            print(f"Error: Input directory not found: {data_dir}")
        else:
            print("Starting batch visualization process...")
            save_visualization_batch(data_dir, output_dir)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback

        traceback.print_exc()