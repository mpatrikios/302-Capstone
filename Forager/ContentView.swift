import SwiftUI
import SwiftCSV
import UIKit

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        let rgbValue = UInt32(hex, radix: 16)
        let r = Double((rgbValue! & 0xFF0000) >> 16) / 255
        let g = Double((rgbValue! & 0x00FF00) >> 8) / 255
        let b = Double(rgbValue! & 0x0000FF) / 255
        self.init(red: r, green: g, blue: b)
    }
}

struct PlantInfo: Edible {
    let id = UUID() //unique id for struct to identify 
    let classNumber: Int
    let description: String
}

struct LoadingCircleView: View {
    @State private var isLoading = false
    
    var body: some View {
        ZStack {
            Image("APPICON")
                .resizable()
                .frame(width: 400, height: 400)
                .rotationEffect(Angle(degrees: isLoading ? 360 : 0))
                .animation(Animation.linear(duration: 1).repeatForever(autoreverses: false), value: self.isLoading)
                .onAppear() {
                    self.isLoading = true
                }
        }
        .padding()
    }
}

//return plant info
func loadCSV() -> PlantInfo? {
    if let path = Bundle.main.path(forResource: "plants", ofType: "csv") {
        do {
            let csv = try CSV(url: URL(fileURLWithPath: path))
            if let row = csv.namedRows.first {
                if let classNumberString = row["classNumber"], let classNumber = Int(classNumberString), let description = row["description"] {
                    return PlantInfo(classNumber: classNumber, description: description)
                }
            }
        } catch {
            print("Error loading CSV: \(error)")
        }
    }
    return nil
}

//LOADING CIRCLE

//struct LoadingCircleView: View {
//    @State private var isLoading = false
//    
//    var body: some View {
//        ZStack {
//            Circle()
//                .stroke(Color(.lightGray), lineWidth: 20)
//            Circle()
//                .trim(from: 0, to: 0.2)
//                .stroke(.tint, lineWidth: 10)
//                .rotationEffect(Angle(degrees: isLoading ? 360 : 0))
//                .animation(Animation.linear(duration: 1).repeatForever(autoreverses: false), value: self.isLoading)
//                .onAppear() {
//                    self.isLoading = true
//                }
//        }
//        //.frame(width:100, height:100)
//        .padding()
//    }
//}

struct ContentView: View {
    @State private var showCamera = false // Boolean to control when to show the camera
    @State private var image: UIImage? // Store the captured image

    var body: some View {
    
        ZStack{
            Color(hex: "#A59F8D") // Set the background color for the entire screen
                           .edgesIgnoringSafeArea(.all) // Ensure the color covers the entire screen
            VStack {
                
                Text("Forager")
                    .font(.system(size: 40, design: .serif))
                    .fontWeight(.semibold)
                                    .foregroundColor(.white)

                                Spacer() // Pushes the content below to the bottom

                    
                // Display the captured image if available, otherwise show a placeholder text
                if let image = image {
                    Spacer()
                    LoadingCircleView()
                    Spacer() // Pushes the content to the bottom
                } else {
                    Text("Snap a pic of your plant!")
                        .foregroundColor(.white)
                        .font(.headline)
                }
                
                // Button to open the camera
                Button(action: {
                    showCamera = true // Show the camera when the button is tapped
                }) {
                    if let image = image {
                        Image(uiImage: image) // Display the captured image
                            .resizable()
                            .scaledToFit()
                            .frame(width: 200, height: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 25)) // Circular shape
                        Button(action: {
                                                    showCamera = true // Show the camera when the button is tapped
                                                }) {
                                                    Text("Take Another")
                                                        .foregroundColor(.white)
                                                        .padding()
                                                        .background(Color(hex: "#808080"))
                                                        .cornerRadius(10)
                                                }
                    } else {
                        Image(systemName: "camera.fill") // Camera icon
                            .font(.system(size: 50)) // Adjust icon size
                            .foregroundColor(.white) // Icon color
                            .padding(100) // Padding inside the grey box
                            .background(Color.gray) // Grey box background
                            .cornerRadius(10) // Rounded corners for the grey box
                            .padding() // Padding around the grey box
                            .clipShape(RoundedRectangle(cornerRadius: 25)) // Circular shape
                    }
                }
                .sheet(isPresented: $showCamera) {
                    CameraView(image: $image) // Present the camera view when the button is tapped
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

