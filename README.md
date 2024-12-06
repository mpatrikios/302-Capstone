# Forager App

Forager is an iOS application designed to identify plants using photos. It leverages CoreML for machine learning-based classification and provides users with plant names and descriptions from a predefined dictionary.

---

## Prerequisites

### Hardware
- An iPhone to run the app.

### Software
- Xcode installed on your macOS system.
- Access to the Forager project code repository.

---

## Setup Instructions

1. **Pull the Code**
   - Clone the Forager app repository from GitHub to your local machine.

2. **Open the Project**
   - Launch Xcode and open the Forager project.

3. **Connect an iPhone**
   - Plug in your iPhone to the computer via USB.
   - Follow Apple's instructions to enable your iPhone as the run destination in Xcode.

4. **Select Destination**
   - In Xcode, select your iPhone as the destination device by clicking the device selection dropdown.

5. **Run the App**
   - Click the "Run" button (triangle icon in the corner of Xcode).
   - The app will be installed and launched on your iPhone.

---

## Using the Forager App

1. **Launch the App**
   - Open the app on your iPhone.
   - During the loading screen, the app will load plant data from a CSV file to create a plant dictionary for identification.

2. **Take or Select a Photo**
   - Press the "Take a Photo" button on the app.
   - Either take a photo of a plant using the camera or select an existing photo from your gallery. The app uses `UIImagePickerController` to handle this process.

3. **Plant Identification**
   - The app processes the image using a trained CoreML model.
   - The image is converted into a `CVPixelBuffer`, which is passed to the model for classification.
   - The model compares the prediction against the plant dictionary:
     - If the plant is identified, the app displays its name and description.
     - If the plant is not in the dictionary, the app displays a message indicating no information is available.

---

## Known Bugs and Limitations

1. **Confidence Score**
   - The app always returns a prediction, even if the confidence score is low.
   - **Future Improvement**: Set a minimum confidence threshold to identify plants accurately.

2. **Model Errors**
   - The machine learning model may misidentify plants due to limitations in training or data quality.

---

## Technical Notes

1. **Core Features**
   - The app uses SwiftUI for UI design and CoreML for machine learning integration.
   - Key imports include:
     - `UIKit`
     - `Foundation`
     - `Vision`
     - `XCTest`
     - `CoreML`

2. **Plant Data**
   - Plant data is stored in a CSV file, which is loaded during app initialization to build a plant dictionary.

3. **Predictions**
   - The app processes images using pixel data and matches predictions to its dictionary for results.

4. **Testing**
   - Test the app by taking or selecting photos of plants.
   - Validate the app's ability to identify plants using its dictionary and note any misclassifications or issues.

---

## Future Enhancements

- Implement a confidence threshold for predictions to improve identification accuracy.
- Enhance the CoreML model with additional training data for better predictions.

## Time Log

| **Date**           | **Activity**                                        | **Members Present**         | **Total Hours** |
|---------------------|----------------------------------------------------|-----------------------------|------------------|
| September 27-29     | Met multiple times to discuss project ideas.       | Anthony, Wyatt, Dylan, Mia, Jake | 2 hours          |
| October 28          | Met after class to break up tasks.                 | Anthony, Wyatt, Dylan, Mia, Jake | 1 hour           |
| November 13         | Made a group presentation and discussed final tasks. | Anthony, Wyatt, Dylan, Mia, Jake | 2 hours          |
| **Total Hours**     |                                                    |                             | **5 hours**      |

---

### Disclaimer
The total hours listed reflect the group meeting times and do not represent the individual time commitment or effort of each group member.



