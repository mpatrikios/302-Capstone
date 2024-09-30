import SwiftUI
import UIKit

// A view that opens the camera and allows the user to capture an image
struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage? // Binding to pass the captured image back to the parent view
    @Environment(\.presentationMode) var presentationMode // To dismiss the camera after capturing the image

    // Coordinator to handle UIImagePickerController delegate methods
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: CameraView

        init(parent: CameraView) {
            self.parent = parent
        }

        // This method is called when an image is picked or captured
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage // Assign the captured image
            }
            parent.presentationMode.wrappedValue.dismiss() // Dismiss the camera view
        }

        // Called when the user cancels
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }

    // Create the UIImagePickerController instance
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera // Set the source type to camera
        return picker
    }

    // Create the coordinator for managing the UIImagePickerController
    func makeCoordinator() -> Coordinator {
        return Coordinator(parent: self)
    }

    // Required method for UIViewControllerRepresentable, but no updates are needed here
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}

