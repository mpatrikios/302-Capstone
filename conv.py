import tensorflow as tf
from tensorflow import keras
import os

def convert_keras_to_h5(keras_path, h5_path):
    """
    Convert a mixed precision .keras model to .h5 format
    
    Parameters:
    keras_path (str): Path to the input .keras file
    h5_path (str): Path for the output .h5 file
    """
    try:
        # Configure mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Load the model
        print(f"Loading model from {keras_path}...")
        model = tf.keras.models.load_model(keras_path, compile=False)
        
        # Save as H5
        print(f"Saving model to {h5_path}...")
        model.save(h5_path, save_format='h5')
        
        print(f"Successfully converted {keras_path} to {h5_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    # Define input and output paths
    keras_path = "best_plant_model.keras"
    h5_path = "plant_classifier.h5"
    
    # Ensure input file exists
    if not os.path.exists(keras_path):
        print(f"Error: Input file {keras_path} not found!")
    else:
        convert_keras_to_h5(keras_path, h5_path)