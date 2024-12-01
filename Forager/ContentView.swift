import SwiftUI
import UIKit
import CoreML
import Foundation

//allows colors from hex value
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

// Loading animation
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

func readCSVFromBundle(filename: String, fileExtension: String = "plants_cleaned.csv") -> [String: String]? {
    // Get the file path from the app's bundle
    guard let filePath = Bundle.main.path(forResource: "plants_cleaned", ofType: "csv") else {
        print("CSV file not found in bundle.")
        return nil
    }
    
    do {
        // Read the content of the CSV file
        let csvData = try String(contentsOfFile: filePath, encoding: .utf8)
        return csvToDictionary(csvData: csvData)
    } catch {
        print("Error reading CSV file: \(error)")
        return nil
    }
}

func csvToDictionary(csvData: String) -> [String: String]? {
    let lines = csvData.split(separator: "\n")
    var plantDictionary = [String: String]()
    
    for line in lines {
        if line.isEmpty { continue }
        
        let components = line.split(separator: ",", maxSplits: 1, omittingEmptySubsequences: false)
        
        if components.count == 2 {
            let plantName = components[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() // Trim spaces and lowercased for case-insensitive match
            let description = components[1].trimmingCharacters(in: .whitespacesAndNewlines)
            plantDictionary[plantName] = description
        }
    }
    for plantName in plantDictionary.keys {
        print("Plant name in dictionary: \(plantName)")
    }

    
    return plantDictionary
}


// Main view
struct ContentView: View {
    @State private var showCamera = false
    @State private var image: UIImage?
    @State private var classificationLabel: String = ""
    @State private var isClassifying: Bool = false
    @State private var classificationError: String?
    @State private var plantData: [String: String] = [:]
    
    // Function to convert UIImage to CVPixelBuffer
    private func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        let width = Int(image.size.width)
        let height = Int(image.size.height)
        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        
        guard let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: CGFloat(height))
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        return buffer
    }
    
    // Function to classify image using Core ML model
    private func classifyImage(_ image: UIImage) {
        isClassifying = true
        classificationError = nil
        
        // Load the CoreML model
        guard let model = try? ForagerML(configuration: MLModelConfiguration()) else {
            classificationError = "Failed to load ML model"
            isClassifying = false
            return
        }
        
        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = pixelBuffer(from: image) else {
            classificationError = "Failed to convert image to pixel buffer"
            isClassifying = false
            return
        }
        
        // Perform prediction
        guard let prediction = try? model.prediction(image: pixelBuffer) else {
            classificationError = "Failed to perform prediction"
            isClassifying = false
            return
        }
        
        // Process the prediction result
        let plantName = prediction.target //model returns the plant name
        if let description = plantData[plantName] {
            classificationLabel = "\(plantName): \(description)"
        } else {
            classificationLabel = "No plant information found for \(plantName)"
        }
        isClassifying = false
    }

    var body: some View {
        ZStack {
            Color(hex: "#A59F8D")
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Text("Forager")
                    .font(.system(size: 40, design: .serif))
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                if let image = image {
                    Spacer()
                    if isClassifying {
                        LoadingCircleView()
                    } else if let error = classificationError {
                        Text(error)
                            .foregroundColor(.red)
                            .padding()
                    } else {
                        Text(classificationLabel)
                            .foregroundColor(.white)
                            .font(.headline)
                            .multilineTextAlignment(.center)
                            .padding()
                    }
                    Spacer()
                } else {
                    Spacer()
                    Text("Snap a pic of your plant!")
                        .foregroundColor(.white)
                        .font(.headline)
                }
                
                Button(action: {
                    showCamera = true
                }) {
                    if let image = image {
                        VStack {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(width: 200, height: 200)
                                .clipShape(RoundedRectangle(cornerRadius: 25))
                            
                            Button(action: {
                                showCamera = true
                            }) {
                                Text("Take Another")
                                    .foregroundColor(.white)
                                    .padding()
                                    .background(Color(hex: "#808080"))
                                    .cornerRadius(10)
                            }
                        }
                    } else {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.white)
                            .padding(100)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding()
                            .clipShape(RoundedRectangle(cornerRadius: 25))
                    }
                }
                .sheet(isPresented: $showCamera) {
                    CameraView(image: $image)
                        .onDisappear {
                            if let capturedImage = image {
                                classifyImage(capturedImage)
                            }
                        }
                }
            }
        }
        .onAppear {
            // Load the plant data from CSV on view load
            if let plantDictionary = readCSVFromBundle(filename: "plants_cleaned") {
                plantData = plantDictionary
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
