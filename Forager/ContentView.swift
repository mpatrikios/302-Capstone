import SwiftUI
import UIKit

struct ContentView: View {
    @State private var showCamera = false // Boolean to control when to show the camera
    @State private var image: UIImage? // Store the captured image

    var body: some View {
        VStack {
            // Display the captured image if available, otherwise show a placeholder text
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 300, height: 300)
            } else {
                Text("No Image Selected")
                    .foregroundColor(.gray)
                    .font(.headline)
            }

            // Button to open the camera
            Button(action: {
                showCamera = true // Show the camera when the button is tapped
            }) {
                Image(systemName: "camera.fill") // Camera icon
                    .font(.system(size: 50)) // Adjust icon size
                    .foregroundColor(.white) // Icon color
                    .padding()
                    .background(Color.blue) // Background color
                    .clipShape(Circle()) // Circular shape
            }
            .sheet(isPresented: $showCamera) {
                CameraView(image: $image) // Present the camera view when the button is tapped
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

