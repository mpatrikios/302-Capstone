import os
import tensorflow as tf
import numpy as np
import coremltools as ct
import tensorflow.keras.backend as K
from datetime import datetime  # Just this one datetime import
from scipy.io import savemat
from tensorflow.keras import Model, regularizers, mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import time
import ssl
import pandas as pd  # Add pandas import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Only show errors, not warnings

ssl._create_default_https_context = ssl._create_unverified_context

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

class PlantClassifier:
    def __init__(self, num_classes=35, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.checkpoint_dir = './training_checkpoints'
        self.batch_size = 16
        self.weight_decay = 0.0001
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = None
        self.base_model = None
        self.total_samples = None

    def create_model(self, num_training_samples):
        self.total_samples = num_training_samples
        steps_per_epoch = num_training_samples // self.batch_size

        # Check for existing checkpoints
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.keras')]
        if checkpoints:
            latest_checkpoint = os.path.join(self.checkpoint_dir, sorted(checkpoints)[-1])
            last_epoch = int(sorted(checkpoints)[-1].split('-')[-1].replace('.keras', ''))
            print(f"\nFound checkpoint: {latest_checkpoint} (Epoch {last_epoch})")
            
            try:
                print("Loading model from checkpoint...")
                self.model = tf.keras.models.load_model(
                    latest_checkpoint,
                    custom_objects={
                        'top_5_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=5)
                    }
                )
                
                # Create and set up base model based on which phase we're in
                self.base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
                
                # Set the appropriate layers to trainable based on epoch
                if last_epoch >= 15:
                    print("Setting up model for final phase (larger set of unfrozen layers)")
                    for layer in self.base_model.layers[-50:]:
                        layer.trainable = True
                    learning_rate = 0.00001
                elif last_epoch >= 10:
                    print("Setting up model for second phase (partially unfrozen)")
                    for layer in self.base_model.layers[-20:]:
                        layer.trainable = True
                    learning_rate = 0.0001
                else:
                    print("Setting up model for initial phase (frozen)")
                    for layer in self.base_model.layers:
                        layer.trainable = False
                    learning_rate = 0.001
                
                # Recompile with appropriate learning rate

                self.model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')],
                    jit_compile=False  # Disable JIT compilation
                )
                
                print(f"Successfully loaded and configured model for epoch {last_epoch + 1}")
                return self.model
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Creating new model instead...")

        # Create new model if no checkpoint exists
        print("Creating new model...")
        self.base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        for layer in self.base_model.layers:
            layer.trainable = False

        # Create the model architecture
        x = self.base_model.output
        x = GlobalAveragePooling2D(name='pooling')(x)
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay), name='dense_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay), name='dense_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay), name='dense_3')(x)
        x = Dropout(0.2, name='dropout_3')(x)
        predictions = Dense(self.num_classes, activation='softmax', dtype='float32', name='predictions')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        
        # Initial compilation
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )

        return self.model

    def predict_batch(self, images, verbose=0):
        try:
            return self.model.predict(images, verbose=verbose)
        except tf.errors.OutOfRangeError:
            print("Warning: Out of range error encountered during prediction. Continuing...")
            return None

    def train(self, train_dataset, validation_dataset, epochs=30):
        """Train the model with proper checkpoint resuming"""
        try:
            os.makedirs('backup_checkpoints', exist_ok=True)
            
            # Determine initial epoch from checkpoints
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.keras')]
            initial_epoch = 0
            if checkpoints:
                last_epoch = max([int(cp.split('-')[-1].replace('.keras', '')) for cp in checkpoints])
                initial_epoch = last_epoch
                print(f"\nResuming training from epoch {initial_epoch}")
            
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.checkpoint_dir, "cp-{epoch:04d}.keras"),
                    save_weights_only=False,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=7,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='best_plant_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.CSVLogger('training_log.csv', append=True),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_begin=lambda epoch, logs: self.unfreeze_layers(epoch)
                )
            ]

            print(f"\nTraining plan:")
            print(f"Starting from epoch: {initial_epoch}")
            print(f"Training until epoch: {epochs}")
            print(f"Epochs remaining: {epochs - initial_epoch}")
            
            # Clear any existing state
            tf.keras.backend.clear_session()
            
            history = self.model.fit(
                train_dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=validation_dataset,
                callbacks=callbacks,
                verbose=1
            )

            return history

        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving backup...")
            backup_path = os.path.join('backup_checkpoints', f'interrupted_model_{int(time.time())}.keras')
            self.model.save(backup_path)
            print(f"Backup saved to: {backup_path}")
            raise

    def create_datasets(self, data_dir, validation_split=0.2):
        print("Setting up data preprocessing...")
        
        IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
        IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])

        def normalize(images, labels):
            images = tf.cast(images, tf.float32) / 255.0
            images = (images - IMAGENET_MEAN) / IMAGENET_STD
            return images, labels

        def augment(images, labels):
            def augment_single_image(image):
                # Random crop and resize
                image = tf.image.resize_with_crop_or_pad(image, 250, 250)
                image = tf.image.random_crop(image, [224, 224, 3])
                
                # Random flips
                image = tf.image.random_flip_left_right(image)
                
                # Color augmentation
                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_contrast(image, 0.8, 1.2)
                image = tf.image.random_saturation(image, 0.8, 1.2)
                image = tf.image.random_hue(image, 0.1)
                
                # Ensure values are in valid range
                image = tf.clip_by_value(image, 0.0, 1.0)
                return image
            
            augmented_images = tf.map_fn(augment_single_image, images, dtype=tf.float32)
            return augmented_images, labels

        try:
            print("Scanning directory for classes...")
            self.class_names = sorted([d for d in os.listdir(data_dir)
                                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
            print(f"Found {len(self.class_names)} classes")

            total_samples = sum(len(os.listdir(os.path.join(data_dir, d))) 
                              for d in self.class_names)
            print(f"Total samples: {total_samples}")

            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                subset="training",
                seed=123,
                image_size=self.input_shape[:2],
                batch_size=self.batch_size,
                label_mode='categorical',
                shuffle=True
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                subset="validation",
                seed=123,
                image_size=self.input_shape[:2],
                batch_size=self.batch_size,
                label_mode='categorical',
                shuffle=True
            )

            AUTOTUNE = tf.data.AUTOTUNE
            
            # Training pipeline
            train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.cache()
            train_ds = train_ds.shuffle(min(total_samples, 10000))
            train_ds = train_ds.prefetch(AUTOTUNE)
            
            # Validation pipeline
            val_ds = val_ds.map(normalize, num_parallel_calls=AUTOTUNE)
            val_ds = val_ds.cache()
            val_ds = val_ds.prefetch(AUTOTUNE)

            self.create_model(int(total_samples * (1 - validation_split)))

            with open('class_names.json', 'w') as f:
                json.dump(self.class_names, f, indent=2)

            return train_ds, val_ds

        except Exception as e:
            print(f"Error in dataset creation: {e}")
            raise

    def unfreeze_layers(self, epoch):
        """Handle layer unfreezing at specific epochs"""
        try:
            if epoch == 10:
                print("\nEpoch 10: Unfreezing final ResNet50 layers...")
                # Store the weights before recompiling
                weights = self.model.get_weights()
                
                # Set trainable layers
                for layer in self.base_model.layers[-20:]:
                    layer.trainable = True
                
                # Recompile model
                self.model = tf.keras.models.clone_model(self.model)
                self.model.compile(
                    optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')],
                    jit_compile=False
                )
                
                # Restore weights
                self.model.set_weights(weights)
                print("Model recompiled with new learning rate (0.0001)")
                
            elif epoch == 15:
                print("\nEpoch 15: Unfreezing more ResNet50 layers...")
                # Store the weights before recompiling
                weights = self.model.get_weights()
                
                # Set trainable layers
                for layer in self.base_model.layers[-50:]:
                    layer.trainable = True
                
                # Recompile model
                self.model = tf.keras.models.clone_model(self.model)
                self.model.compile(
                    optimizer=Adam(learning_rate=0.00001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')],
                    jit_compile=False
                )
                
                # Restore weights
                self.model.set_weights(weights)
                print("Model recompiled with final learning rate (0.00001)")
                
        except Exception as e:
            print(f"Error in unfreeze_layers: {e}")
            print("Attempting recovery by recreating optimizer...")
            try:
                # Simple recompilation without changing architecture
                current_weights = self.model.get_weights()
                self.model.compile(
                    optimizer=Adam(learning_rate=0.00001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')],
                    jit_compile=False
                )
                self.model.set_weights(current_weights)
                print("Recovery successful")
            except Exception as e:
                print(f"Recovery failed: {e}")
                print("Continuing with current configuration...")

    def evaluate_model(self, validation_dataset):
        print("\nEvaluating model performance...")
        true_labels = []
        predictions = []

        try:
            for images, labels in validation_dataset:
                true_labels.extend(np.argmax(labels, axis=1))
                preds = self.model.predict(images, verbose=0)
                predictions.extend(np.argmax(preds, axis=1))

            # Handle zero division in classification report
            report = classification_report(
                true_labels, 
                predictions, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0  # Add this parameter
            )

            metrics = {
                'class_names': self.class_names,
                'metrics': report
            }

            with open('model_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            # Create confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(20, 20))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'confusion_matrix_{timestamp}.png')
            plt.close()

        except Exception as e:
            print(f"Error in model evaluation: {e}")

    def plot_training_history(self, history):
        try:
            # Try to load existing history from CSV
            existing_history = None
            if os.path.exists('training_log.csv'):
                try:
                    import pandas as pd
                    existing_history = pd.read_csv('training_log.csv')
                except Exception as e:
                    print(f"Could not load existing history: {e}")

            plt.figure(figsize=(15, 5))
            
            # Plot Accuracy
            plt.subplot(1, 2, 1)
            if existing_history is not None:
                plt.plot(existing_history['epoch'], existing_history['accuracy'], 
                        color='blue', alpha=0.7, label='Previous Training Accuracy')
                plt.plot(existing_history['epoch'], existing_history['val_accuracy'], 
                        color='orange', alpha=0.7, label='Previous Validation Accuracy')
                
                # Plot current history continuing from previous
                start_epoch = len(existing_history)
                epochs = range(start_epoch, start_epoch + len(history.history['accuracy']))
                plt.plot(epochs, history.history['accuracy'], 
                        color='blue', linestyle='--', label='Current Training Accuracy')
                plt.plot(epochs, history.history['val_accuracy'], 
                        color='orange', linestyle='--', label='Current Validation Accuracy')
            else:
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)

            # Plot Loss
            plt.subplot(1, 2, 2)
            if existing_history is not None:
                plt.plot(existing_history['epoch'], existing_history['loss'], 
                        color='blue', alpha=0.7, label='Previous Training Loss')
                plt.plot(existing_history['epoch'], existing_history['val_loss'], 
                        color='orange', alpha=0.7, label='Previous Validation Loss')
                
                start_epoch = len(existing_history)
                epochs = range(start_epoch, start_epoch + len(history.history['loss']))
                plt.plot(epochs, history.history['loss'], 
                        color='blue', linestyle='--', label='Current Training Loss')
                plt.plot(epochs, history.history['val_loss'], 
                        color='orange', linestyle='--', label='Current Validation Loss')
            else:
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
            
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'training_history_{timestamp}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved as {save_path}")
            plt.close()

            # Also save the raw history data
            try:
                if existing_history is not None:
                    # Combine old and new history
                    new_history = pd.DataFrame({
                        'epoch': range(start_epoch, start_epoch + len(history.history['accuracy'])),
                        'accuracy': history.history['accuracy'],
                        'val_accuracy': history.history['val_accuracy'],
                        'loss': history.history['loss'],
                        'val_loss': history.history['val_loss']
                    })
                    combined_history = pd.concat([existing_history, new_history], ignore_index=True)
                    combined_history.to_csv(f'complete_training_history_{timestamp}.csv', index=False)
                else:
                    pd.DataFrame({
                        'epoch': range(len(history.history['accuracy'])),
                        'accuracy': history.history['accuracy'],
                        'val_accuracy': history.history['val_accuracy'],
                        'loss': history.history['loss'],
                        'val_loss': history.history['val_loss']
                    }).to_csv(f'complete_training_history_{timestamp}.csv', index=False)
            except Exception as e:
                print(f"Error saving history data: {e}")

        except Exception as e:
            print(f"Error plotting training history: {e}")
            # Create a simple fallback plot if something goes wrong
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history_fallback.png')
            plt.close()
def main():
    print("Starting Wild Plants Classifier Training\n")
    print("This process will:")
    print("1. Download the dataset")
    print("2. Train the model (this may take several hours)")
    print("3. Save training progress and metrics")
    print("4. Save the final model for CoreML conversion\n")
    
    print("Downloading dataset...")
    try:
        dataset_path = kagglehub.dataset_download("ryanpartridge01/wild-edible-plants")
        data_dir = os.path.join(dataset_path, "dataset", "resized")
        print(f"Dataset downloaded to: {data_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        exit(1)

    print(f"\nInitializing classifier...")
    classifier = PlantClassifier(num_classes=35)

    print("\nPreparing datasets...")
    train_dataset, validation_dataset = classifier.create_datasets(data_dir)

    print("\nStarting training...")
    history = classifier.train(train_dataset, validation_dataset)

    #print("\nSaving final model...")
    #classifier.model.save('plant_classifier_for_coreml', save_format="keras")
    #classifier.model.save('plant_classifier_for_coreml.keras')
    #print("\nModel saved as 'plant_classifier_for_coreml.keras'")
    print("\nSaving final model in H5 format...")
    try:
        classifier.model.save('plant_classifier.h5', save_format='h5')
        print("\nModel saved as 'plant_classifier.h5'")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        # Fallback save attempt
        print("Attempting alternative save method...")
        try:
            tf.keras.models.save_model(
                classifier.model,
                'plant_classifier.h5',
                save_format='h5',
                include_optimizer=False
            )
            print("Model saved successfully using alternative method")
        except Exception as e2:
            print(f"Failed to save model: {str(e2)}")

if __name__ == "__main__":
    main()