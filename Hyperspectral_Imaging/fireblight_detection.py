import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import os
import logging
import traceback
import time
from sklearn.metrics import precision_score
from scipy.ndimage import rotate
from data_processing_function import load_and_preprocess_data
from feature_extraction import extract_all_features
from gpu_utils import gpu_manager
from validation_utils import (
    validate_data_quality,
    cross_validate_indicators,
    filter_false_positives
)


class FireBlightDetectionPipeline:
    def __init__(self, input_shape=(224, 224, 3), model_path=None):
        self.input_shape = input_shape
        self.class_labels = ['Healthy', 'Early Stage', 'Moderate', 'Severe', 'Critical']
        self.model = self._build_model() if model_path is None else tf.keras.models.load_model(model_path)
        self.history = None

    def _build_model(self):
        """Build EfficientNetV2-L model with custom layers"""
        base_model = efficientnet_v2.EfficientNetV2L(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )

        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(len(self.class_labels), activation='softmax')(x)

        return Model(inputs=base_model.input, outputs=outputs)

    def split_dataset(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into training, validation, and test sets with handling for small datasets
        """
        unique_classes, class_counts = np.unique(y, return_counts=1)
        min_samples = np.min(class_counts)

        if min_samples < 2:
            # If we have too few samples for stratification, use simple splitting
            logging.warning(
                f"Found class(es) with only {min_samples} sample(s). Using simple random split instead of stratified split.")

            # First split: separate test set
            if len(X) < 5:  # Very small dataset
                logging.warning("Dataset too small for proper splitting. Using same data for train/val/test.")
                return (X, y), (X, y), (X, y)

            try:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Second split: separate validation set
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=42
                )

            except ValueError as e:
                logging.warning(f"Splitting failed: {str(e)}. Using simple array splits.")
                # Manual splitting for very small datasets
                total = len(X)
                test_idx = int(total * 0.8)  # Use 80% for training/validation
                val_idx = int(test_idx * 0.8)  # Use 80% of remaining for training

                X_train = X[:val_idx]
                y_train = y[:val_idx]
                X_val = X[val_idx:test_idx]
                y_val = y[val_idx:test_idx]
                X_test = X[test_idx:]
                y_test = y[test_idx:]
        else:
            # Regular stratified split if we have enough samples
            try:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )

                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
                )
            except ValueError as e:
                logging.warning(f"Stratified split failed: {str(e)}. Falling back to simple split.")
                return self.split_dataset(X, y, test_size, val_size)  # Recursive call without stratification

        logging.info(
            f"Dataset split complete - Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)} samples")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_feature_matrix(self, hdr_paths, days_info):
        """Build feature matrix from hyperspectral data"""
        X = []
        y = []
        metadata = []

        for hdr_path, day in zip(hdr_paths, days_info):
            try:
                processed_data, wavelengths = load_and_preprocess_data(hdr_path)

                if processed_data is None:
                    logging.warning(f"Skipping {hdr_path}: Failed to load or process data")
                    continue

                # Extract features
                features, disease_mask, labeled_regions = extract_all_features(processed_data, wavelengths)

                if features is None:
                    logging.warning(f"Skipping {hdr_path}: Failed to extract features")
                    continue

                # Get the disease indices and patterns
                disease_indices = features['disease_indices']
                disease_patterns = features['disease_patterns']

                # Get base shape from NDVI
                base_shape = disease_indices['disease_NDVI'].shape

                # Create arrays with consistent shapes
                ndvi = disease_indices['disease_NDVI']
                green_nir = disease_indices['green_NIR_ratio']

                # Create disease severity map with same shape as other indices
                disease_severity = np.full(base_shape, disease_patterns['disease_severity'])

                # Verify shapes match before stacking
                if not (ndvi.shape == green_nir.shape == disease_severity.shape):
                    logging.error(f"Shape mismatch in {hdr_path}:")
                    logging.error(f"NDVI shape: {ndvi.shape}")
                    logging.error(f"Green-NIR shape: {green_nir.shape}")
                    logging.error(f"Disease severity shape: {disease_severity.shape}")
                    continue

                # Normalize each component individually
                def normalize_array(arr):
                    min_val = np.min(arr)
                    max_val = np.max(arr)
                    return (arr - min_val) / (max_val - min_val + 1e-8)

                ndvi_norm = normalize_array(ndvi)
                green_nir_norm = normalize_array(green_nir)
                severity_norm = normalize_array(disease_severity)

                # Stack the normalized arrays
                image_data = np.stack([
                    ndvi_norm,
                    green_nir_norm,
                    severity_norm
                ], axis=-1)

                # Ensure the image has the correct dimensions for the model
                image_data = tf.image.resize(image_data, (224, 224))

                # Calculate severity score
                severity_score = self._calculate_severity_score(features, day)

                X.append(image_data)
                y.append(severity_score)

                # Enhanced metadata including shapes and normalization info
                metadata.append({
                    'file': str(hdr_path),
                    'day': day,
                    'features': {
                        'disease_severity': float(features['disease_patterns']['disease_severity']),
                        'affected_area_ratio': float(features['disease_patterns']['affected_area_ratio']),
                        'lesion_count': int(features['disease_patterns']['lesion_count']),
                        'avg_lesion_size': float(features['disease_patterns']['avg_lesion_size']),
                        'pattern_uniformity': float(features['disease_patterns']['pattern_uniformity'])
                    },
                    'array_shapes': {
                        'original_ndvi': ndvi.shape,
                        'original_green_nir': green_nir.shape,
                        'final_image': image_data.shape
                    },
                    'normalization': {
                        'ndvi_range': [float(np.min(ndvi)), float(np.max(ndvi))],
                        'green_nir_range': [float(np.min(green_nir)), float(np.max(green_nir))],
                        'severity_range': [float(np.min(disease_severity)), float(np.max(disease_severity))]
                    }
                })

            except Exception as e:
                logging.error(f"Error processing file {hdr_path}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        if not X:
            raise ValueError("No valid samples were created")

        return np.array(X), np.array(y), metadata

    def _calculate_severity_score(self, features, day):
        """Calculate disease severity class with enhanced metrics"""
        disease_patterns = features['disease_patterns']

        # Enhanced severity calculation including new metrics
        severity = (
                0.35 * disease_patterns['disease_severity'] +
                0.25 * disease_patterns['affected_area_ratio'] +
                0.15 * (1 - np.exp(-day / 5)) +
                0.15 * (disease_patterns['lesion_count'] / 100 if disease_patterns['lesion_count'] < 100 else 1.0) +
                0.10 * disease_patterns['pattern_uniformity']
        )

        # Adjust severity based on lesion size if available
        if disease_patterns['lesion_count'] > 0:
            avg_lesion_size_factor = min(disease_patterns['avg_lesion_size'] / 1000, 1.0)
            severity = 0.9 * severity + 0.1 * avg_lesion_size_factor

        return np.digitize(severity, bins=[0.2, 0.4, 0.6, 0.8]) - 1

    # Data Augmentation Functions
    def rotate_spectral_data(self, image, angles=[90, 180, 270]):
        """Create rotated versions of spectral data"""
        augmented = [image]
        for angle in angles:
            rotated = rotate(image, angle, axes=(0, 1), reshape=False)
            augmented.append(rotated)
        return augmented

    def add_noise(self, image, noise_factor=0.05):
        """Add Gaussian noise to image"""
        noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)

    def simulate_conditions(self, image, brightness_range=(-0.2, 0.2)):
        """Simulate different lighting conditions"""
        brightness = np.random.uniform(*brightness_range)
        adjusted = image + brightness
        return np.clip(adjusted, 0, 1)

    def generate_synthetic_samples(self, X, y, augmentation_factor=2):
        """Generate synthetic training examples using augmentation"""
        X_augmented = []
        y_augmented = []

        for image, label in zip(X, y):
            # Original sample
            X_augmented.append(image)
            y_augmented.append(label)

            # Rotated samples
            rotated = self.rotate_spectral_data(image)
            X_augmented.extend(rotated[1:])  # Exclude original
            y_augmented.extend([label] * (len(rotated) - 1))

            # Noisy samples
            noisy = self.add_noise(image)
            X_augmented.append(noisy)
            y_augmented.append(label)

            # Different lighting conditions
            adjusted = self.simulate_conditions(image)
            X_augmented.append(adjusted)
            y_augmented.append(label)

        return np.array(X_augmented), np.array(y_augmented)

    def train_model(self, train_data, val_data, epochs=50, batch_size=32, callbacks=None):
        """
        Train the model with the provided data
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Log training parameters
        logging.info(f"""
        Training Parameters:
        - Input shape: {X_train[0].shape}
        - Training samples: {len(X_train)}
        - Validation samples: {len(X_val)}
        - Epochs: {epochs}
        - Batch size: {batch_size}
        - Learning rate: 1e-4
        """)

        # Data augmentation for training set
        logging.info("Generating augmented training data...")
        X_train_aug, y_train_aug = self.generate_synthetic_samples(X_train, y_train)
        logging.info(f"Augmented training set size: {len(X_train_aug)} samples")

        # Calculate class weights
        unique_classes = np.unique(y_train_aug)
        class_counts = np.bincount(y_train_aug)
        total_samples = len(y_train_aug)

        class_weights = {}
        for cls in unique_classes:
            class_weights[cls] = total_samples / (len(unique_classes) * class_counts[cls])

        logging.info(f"Class weights: {class_weights}")

        # Compile model with correct metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False
            ),
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseCategoricalAccuracy(name='categorical_accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy')
            ]
        )

        # Use provided callbacks or default ones
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    verbose=1,
                    min_lr=1e-6
                )
            ]

        # Train the model
        logging.info("Starting model training...")
        self.history = self.model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights  # Use calculated class weights
        )

        logging.info("Training completed!")

        # Log final training metrics
        final_epoch = len(self.history.history['loss']) - 1
        logging.info(f"""
        Final Training Metrics:
        - Loss: {self.history.history['loss'][final_epoch]:.4f}
        - Accuracy: {self.history.history['accuracy'][final_epoch]:.4f}
        - Validation Loss: {self.history.history['val_loss'][final_epoch]:.4f}
        - Validation Accuracy: {self.history.history['val_accuracy'][final_epoch]:.4f}
        """)

        return self.history

    def validate_model(self, val_data):
        """Perform model validation"""
        X_val, y_val = val_data
        val_predictions = self.model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)

        return self.calculate_metrics(y_val, val_pred_classes)

    def test_model(self, test_data):
        """Evaluate model performance on test set"""
        X_test, y_test = test_data
        test_predictions = self.model.predict(X_test)
        test_pred_classes = np.argmax(test_predictions, axis=1)

        return self.calculate_metrics(y_test, test_pred_classes)

    def predict_quality(self, hdr_path, day=None):
        """Make predictions on new samples with enhanced feature analysis and validation."""
        try:
            # Load and preprocess data with quality validation
            processed_data, wavelengths = load_and_preprocess_data(hdr_path)
            if processed_data is None or wavelengths is None:
                raise ValueError("Failed to load or preprocess data")

            # Validate data quality
            quality_metrics = validate_data_quality(processed_data, wavelengths)
            if not quality_metrics['valid_data']:
                logging.warning(f"Data quality issues detected: {quality_metrics['issues']}")

            # Extract features and disease patterns
            features, disease_mask, labeled_regions = extract_all_features(processed_data, wavelengths)
            if features is None:
                raise ValueError("Failed to extract features")

            # Get disease indices and patterns
            disease_indices = features['disease_indices']
            disease_patterns = features['disease_patterns']

            # Cross-validate indicators
            validation_score, confidence_metrics = cross_validate_indicators(
                disease_indices['disease_NDVI'],
                disease_indices['green_NIR_ratio'],
                features.get('texture_features', {})
            )

            # Create input image data for model
            image_data = np.stack([
                disease_indices['disease_NDVI'],
                disease_indices['green_NIR_ratio'],
                np.full_like(disease_indices['disease_NDVI'],
                             disease_patterns['disease_severity'])
            ], axis=-1)

            # Normalize the image data
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8)
            image_data = tf.image.resize(image_data, (224, 224))

            # Make prediction
            prediction = self.model.predict(np.expand_dims(image_data, axis=0))
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])

            # Prepare initial prediction result
            result = {
                'predicted_stage': self.class_labels[predicted_class],
                'confidence': confidence,
                'raw_predictions': prediction[0].tolist(),
                'features': {
                    'disease_patterns': {
                        'disease_severity': float(disease_patterns['disease_severity']),
                        'affected_area_ratio': float(disease_patterns['affected_area_ratio']),
                        'lesion_count': int(disease_patterns['lesion_count']),
                        'avg_lesion_size': float(disease_patterns['avg_lesion_size']),
                        'pattern_uniformity': float(disease_patterns['pattern_uniformity'])
                    },
                    'lesion_analysis': {
                        'total_lesions': int(disease_patterns['lesion_count']),
                        'avg_lesion_size': float(disease_patterns['avg_lesion_size']),
                        'pattern_uniformity': float(disease_patterns['pattern_uniformity'])
                    },
                    'affected_area': {
                        'ratio': float(disease_patterns['affected_area_ratio']),
                        'severity': float(disease_patterns['disease_severity'])
                    },
                    'disease_indices': disease_indices
                },
                'validation_metrics': {
                    'validation_score': validation_score,
                    'confidence_metrics': confidence_metrics,
                    'quality_metrics': quality_metrics
                },
                'visualization_data': {
                    'disease_mask': disease_mask.tolist(),
                    'labeled_regions': labeled_regions.tolist()
                }
            }

            # Add temporal information if available
            if day is not None:
                result['temporal_info'] = {'day': int(day)}

            # Filter false positives
            result = filter_false_positives(result)

            # Adjust confidence based on validation score
            result['confidence'] *= validation_score

            # Log prediction results
            logging.info(f"Prediction made - Stage: {result['predicted_stage']}, "
                         f"Confidence: {result['confidence']:.3f}, "
                         f"Validation Score: {validation_score:.3f}")

            return result

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    # Metric Calculation Functions
    def calculate_accuracy(self, y_true, y_pred):
        """Compute model accuracy metrics"""
        return {
            'accuracy': float(np.mean(y_true == y_pred)),
            'per_class_accuracy': [
                float(np.mean((y_true == i) & (y_pred == i)))
                for i in range(len(self.class_labels))
            ]
        }

    def evaluate_precision(self, y_true, y_pred):
        """Calculate precision scores with zero division handling"""
        return {
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'weighted_precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'per_class_precision': precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        }

    def assess_recall(self, y_true, y_pred):
        """Calculate recall scores"""
        return {
            'macro_recall': float(recall_score(y_true, y_pred, average='macro')),
            'weighted_recall': float(recall_score(y_true, y_pred, average='weighted')),
            'per_class_recall': recall_score(y_true, y_pred, average=None).tolist()
        }

    def calculate_f1_score(self, y_true, y_pred):
        """Compute F1 scores"""
        return {
            'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
            'per_class_f1': f1_score(y_true, y_pred, average=None).tolist()
        }

    def generate_confusion_matrix(self, y_true, y_pred, output_path=None):
        """Create confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_labels,
                    yticklabels=self.class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if output_path:
            plt.savefig(output_path)
            plt.close()
            return output_path
        else:
            plt.close()
            return cm.tolist()

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        return {
            'accuracy': self.calculate_accuracy(y_true, y_pred),
            'precision': self.evaluate_precision(y_true, y_pred),
            'recall': self.assess_recall(y_true, y_pred),
            'f1_score': self.calculate_f1_score(y_true, y_pred),
            'confusion_matrix': self.generate_confusion_matrix(y_true, y_pred)
        }

    def generate_performance_report(self, metrics, output_path=None):
        """Create comprehensive performance report with proper JSON serialization"""
        try:
            def serialize_numpy(obj):
                """Handle numpy and tensor types for JSON serialization"""
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif hasattr(obj, 'numpy'):  # For tensorflow tensors
                    return obj.numpy().tolist()
                elif hasattr(obj, 'shape'):  # For tensor shapes
                    return list(obj)
                return obj

            report = {
                'timestamp': datetime.now().isoformat(),
                'model_summary': {
                    'total_params': int(self.model.count_params()),
                    'input_shape': list(self.input_shape),
                    'num_classes': len(self.class_labels)
                },
                'metrics': metrics,
                'class_labels': self.class_labels
            }

            if self.history:
                report['training_history'] = {
                    'accuracy': [float(x) for x in self.history.history['accuracy']],
                    'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
                    'loss': [float(x) for x in self.history.history['loss']],
                    'val_loss': [float(x) for x in self.history.history['val_loss']]
                }

            # Deep copy and convert all numpy/tensor types
            def convert_dict(d):
                """Recursively convert dictionary values to JSON serializable types"""
                result = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        result[k] = convert_dict(v)
                    elif isinstance(v, (list, tuple)):
                        result[k] = [serialize_numpy(x) for x in v]
                    else:
                        result[k] = serialize_numpy(v)
                return result

            report = convert_dict(report)

            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=4)

            return report

        except Exception as e:
            logging.error(f"Error generating performance report: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def generate_analysis_visualization(self, image_data, disease_mask, labeled_regions, predictions=None):
        """Generate comprehensive visualization of analysis results"""
        plt.figure(figsize=(20, 10))

        # Original image
        plt.subplot(231)
        plt.imshow(image_data)
        plt.title('Original Image')
        plt.colorbar()

        # Disease mask
        plt.subplot(232)
        plt.imshow(disease_mask, cmap='RdYlBu_r')
        plt.title('Disease Detection Mask')
        plt.colorbar(label='Disease Presence')

        # Lesion identification
        plt.subplot(233)
        plt.imshow(labeled_regions, cmap='nipy_spectral')
        plt.title(f'Detected Lesions\nCount: {len(np.unique(labeled_regions)) - 1}')
        plt.colorbar(label='Lesion ID')

        # Disease severity heatmap
        plt.subplot(234)
        severity_map = disease_mask * np.ones_like(labeled_regions)
        plt.imshow(severity_map, cmap='hot')
        plt.title('Disease Severity Heatmap')
        plt.colorbar(label='Severity')

        # If predictions are provided, show confidence
        if predictions is not None:
            plt.subplot(235)
            confidence_plot = plt.bar(self.class_labels, predictions)
            plt.title('Model Confidence by Class')
            plt.xticks(rotation=45)
            plt.ylabel('Confidence')

        plt.tight_layout()
        return plt.gcf()

    def export_results(self, results, output_dir):
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, 'model.keras')
        self.model.save(model_path)

        # Save performance report
        report_path = os.path.join(output_dir, 'performance_report.json')
        self.generate_performance_report(results, report_path)

        # Export predictions and features to CSV
        predictions_df = pd.DataFrame()
        if 'test_data' in results and 'test_predictions' in results:
            y_test = results['test_data'][1]
            test_predictions = results['test_predictions']

            predictions_df['true_label'] = [self.class_labels[y] for y in y_test]
            predictions_df['predicted_label'] = [self.class_labels[np.argmax(p)] for p in test_predictions]
            predictions_df['confidence'] = [np.max(p) for p in test_predictions]

            if 'metadata' in results:
                for idx, meta in enumerate(results['metadata']):
                    predictions_df.loc[idx, 'file'] = meta['file']
                    predictions_df.loc[idx, 'day'] = meta['day']
                    predictions_df.loc[idx, 'disease_severity'] = meta['features']['disease_severity']
                    predictions_df.loc[idx, 'affected_area_ratio'] = meta['features']['affected_area_ratio']
                    predictions_df.loc[idx, 'lesion_count'] = meta['features']['lesion_count']

            predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

        # Save training history plots
        if self.history:
            plt.figure(figsize=(15, 5))

            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Training')
            plt.plot(self.history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Training')
            plt.plot(self.history.history['val_loss'], label='Validation')
            plt.title('Model Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_history.png'))
            plt.close()

            # Save history to CSV
            history_df = pd.DataFrame(self.history.history)
            history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

        # Save confusion matrix
        if 'test_metrics' in results:
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            self.generate_confusion_matrix(
                results['test_data'][1],
                np.argmax(results['test_predictions'], axis=1),
                cm_path
            )

        # Save model architecture visualization
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=os.path.join(output_dir, 'model_architecture.png'),
                show_shapes=True,
                show_layer_names=True
            )
        except Exception as e:
            print(f"Could not save model architecture visualization: {str(e)}")

        return output_dir

    def generate_reports(self, results, output_dir):
        """Create detailed analysis reports"""
        os.makedirs(output_dir, exist_ok=True)

        # Generate detailed HTML report
        html_report = f"""
        <html>
        <head>
            <title>Fire Blight Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; padding: 20px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                h1, h2 {{ color: #2c3e50; }}
                .metric-card {{ 
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{ 
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ color: #2980b9; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Fire Blight Detection Analysis Report</h1>
                <div class="metric-card">
                    <h2>Executive Summary</h2>
                    <p><strong>Model Performance:</strong></p>
                    <ul>
                        <li>Overall Accuracy: <span class="highlight">
                            {results['test_metrics']['accuracy']['accuracy']:.3f}</span></li>
                        <li>Macro F1 Score: <span class="highlight">
                            {results['test_metrics']['f1_score']['macro_f1']:.3f}</span></li>
                    </ul>
                </div>

                <div class="metric-card">
                    <h2>Detailed Class Performance</h2>
                    <table>
                        <tr>
                            <th>Disease Stage</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                        </tr>
        """

        # Add per-class metrics
        for i, class_label in enumerate(self.class_labels):
            html_report += f"""
                        <tr>
                            <td>{class_label}</td>
                            <td>{results['test_metrics']['precision']['per_class_precision'][i]:.3f}</td>
                            <td>{results['test_metrics']['recall']['per_class_recall'][i]:.3f}</td>
                            <td>{results['test_metrics']['f1_score']['per_class_f1'][i]:.3f}</td>
                        </tr>
            """

        # Add model details
        html_report += f"""
                    </table>
                </div>

                <div class="metric-card">
                    <h2>Model Information</h2>
                    <ul>
                        <li>Input Shape: {self.input_shape}</li>
                        <li>Number of Classes: {len(self.class_labels)}</li>
                        <li>Total Parameters: {self.model.count_params():,}</li>
                    </ul>
                </div>

                <div class="metric-card">
                    <h2>Generated Files</h2>
                    <ul>
                        <li>Model Weights: model.h5</li>
                        <li>Training History: training_history.csv</li>
                        <li>Predictions: predictions.csv</li>
                        <li>Confusion Matrix: confusion_matrix.png</li>
                        <li>Training Plots: training_history.png</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
            f.write(html_report)

        # Generate summary text report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Fire Blight Detection Model - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Model Performance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Accuracy: {results['test_metrics']['accuracy']['accuracy']:.3f}\n")
            f.write(f"Macro F1 Score: {results['test_metrics']['f1_score']['macro_f1']:.3f}\n\n")

            f.write("Per-Class Performance:\n")
            f.write("-" * 20 + "\n")
            for i, class_label in enumerate(self.class_labels):
                f.write(f"\n{class_label}:\n")
                f.write(f"  Precision: {results['test_metrics']['precision']['per_class_precision'][i]:.3f}\n")
                f.write(f"  Recall: {results['test_metrics']['recall']['per_class_recall'][i]:.3f}\n")
                f.write(f"  F1 Score: {results['test_metrics']['f1_score']['per_class_f1'][i]:.3f}\n")


def main():
    """Main execution function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Set up paths
        current_dir = Path.cwd()
        data_dir = current_dir / "hyperspectral_data"
        output_dir = current_dir / "results"
        output_dir.mkdir(exist_ok=True)

        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        else:
            logger.info("No GPUs found. Running on CPU.")

        # Get all HDR files and sort them
        logger.info("Scanning for hyperspectral data files...")
        hdr_files = sorted(data_dir.glob("REFLECTANCE_*.hdr"))
        if not hdr_files:
            raise FileNotFoundError("No .hdr files found in the data directory")
        logger.info(f"Found {len(hdr_files)} hyperspectral data files")

        # Extract temporal information from filenames
        days_info = []
        for hdr_file in hdr_files:
            try:
                # Extract number from REFLECTANCE_XXXX format
                number = int(hdr_file.stem.split('_')[1])
                days_info.append(number)
            except (IndexError, ValueError):
                logger.warning(f"Could not extract number from {hdr_file.name}")
                days_info.append(0)

        # Normalize days to start from 0
        if days_info:
            min_day = min(days_info)
            days_info = [d - min_day for d in days_info]
            logger.info(f"Normalized temporal range: 0 to {max(days_info)} units")

        # Initialize the pipeline
        logger.info("Initializing FireBlight Detection Pipeline...")
        pipeline = FireBlightDetectionPipeline()

        # Create feature matrix
        logger.info("Creating feature matrix from hyperspectral data...")
        X, y, metadata = pipeline.create_feature_matrix(hdr_files, days_info)
        logger.info(f"Created feature matrix with {len(X)} samples")

        # Split dataset
        logger.info("Splitting dataset into train/validation/test sets...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.split_dataset(X, y)
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                str(output_dir / 'training_log.csv'),
                separator=',',
                append=False
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(output_dir / 'logs'),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=0
            )
        ]

        # Train model
        logger.info("Training model...")
        history = pipeline.train_model(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            callbacks=callbacks
        )
        logger.info("Model training completed")

        # Evaluate model
        logger.info("Evaluating model performance...")
        test_predictions = pipeline.model.predict(X_test)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_metrics = pipeline.calculate_metrics(y_test, test_pred_classes)

        # Compile results
        results = {
            'metadata': metadata,
            'test_data': (X_test, y_test),
            'test_predictions': test_predictions,
            'test_metrics': test_metrics,
            'training_history': history.history,
            'temporal_info': {
                'original_numbers': sorted(set([int(f.stem.split('_')[1]) for f in hdr_files])),
                'normalized_days': sorted(set(days_info))
            }
        }

        # Export results and generate reports
        logger.info("Generating reports and exporting results...")
        pipeline.export_results(results, output_dir)
        pipeline.generate_reports(results, output_dir)

        # Save final model
        final_model_path = output_dir / 'final_model.keras'
        pipeline.model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # Print final performance summary
        logger.info(f"""
        Training and evaluation completed successfully!

        Model Performance Summary:
        ------------------------
        Accuracy: {test_metrics['accuracy']['accuracy']:.3f}
        Macro F1 Score: {test_metrics['f1_score']['macro_f1']:.3f}

        Per-Class Performance:
        --------------------
        {json.dumps(test_metrics['precision']['per_class_precision'], indent=2)}

        Results and reports have been saved to: {output_dir}
        """)

        return pipeline, results

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    trained_pipeline, results = main()