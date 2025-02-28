//
//  ContentView.swift
//  YOLOv8SwiftUI
//
//  Created by Jin on 2024-03-19.
//

import SwiftUI
import os.log
import PhotosUI
import UIKit
import Vision
import CoreML

struct ContentView: View {
    @StateObject private var model = DataModel()
    @State private var selectedItem: PhotosPickerItem?
    
    var body: some View {
        VStack {
            GeometryReader { geometry in
                if let previewImage = model.previewImage {
                    previewImage
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .overlay {
                            GeometryReader { (geometry: GeometryProxy) in
                                ForEach(model.recognizedTexts) { text in
                                    BoundingBox(imageViewGeometry: geometry,
                                              label: text.text,
                                              rect: text.boundingBox,
                                              color: Color.green,
                                              hideLabel: false)
                                }
                                
                                if let densestRegion = model.densestRegion {
                                    BoundingBox(imageViewGeometry: geometry,
                                              label: "文本密度最高區域",
                                              rect: densestRegion,
                                              color: Color.red,
                                              hideLabel: false)
                                }
                            }
                        }
                } else {
                    Color.black
                }
            }
            
            PhotosPicker(selection: $selectedItem,
                        matching: .images) {
                Text("選擇圖片")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .onChange(of: selectedItem) { newItem in
                Task {
                    await model.processPickerResult(newItem)
                }
            }
        }
        .ignoresSafeArea(.all, edges: .top)
    }

    // Draw a bounding box around the recognized object
    private func BoundingBox(imageViewGeometry: GeometryProxy, label: String, rect: CGRect, color: Color, hideLabel: Bool) -> some View {
        let cgRect = self.denormalize(imageViewSize: imageViewGeometry.size, normalizedCGRect: rect)
            return Rectangle().path(in: cgRect)
                .stroke(color, lineWidth: 2.0)
                .overlay {
                    hideLabel ? nil : Text(label)
                        .foregroundColor(.white)
                        .background(Color.green)
                        .position(x: cgRect.minX, y: cgRect.minY)
                }
    }
    
    // Convert the normalized CGRect to a denormalized CGRect
    private func denormalize(imageViewSize: CGSize, normalizedCGRect: CGRect) -> CGRect {
        let imageViewWidth = imageViewSize.width
        let imageViewHeight = imageViewSize.height

        // Flip the Y coordinate, because the Vision framework uses a coordinate system with the origin in the bottom-left corner, while the SwiftUI uses a coordinate system with the origin in the top-left corner.
        let flippedY = 1.0 - normalizedCGRect.maxY
        return CGRect(x: normalizedCGRect.minX * imageViewWidth, y: flippedY * imageViewHeight, width: normalizedCGRect.width * imageViewWidth, height: normalizedCGRect.height * imageViewHeight)
    }
    
    @MainActor class DataModel : ObservableObject {
        @Published var previewImage: Image?
        @Published var recognizedTexts: [RecognizedText] = []
        @Published var densestRegion: CGRect?
        
        // 密度分析配置
        struct DensityConfig {
            var sections: Int           // 水平區域數量
            var minimumDensity: Int    // 最小文字數量閾值
            
            static let `default` = DensityConfig(
                sections: 10,
                minimumDensity: 1
            )
        }
        
        var densityConfig: DensityConfig = .default
        
        init(densityConfig: DensityConfig = .default) {
            self.densityConfig = densityConfig
        }
        
        // 更新密度分析配置
        func updateDensityConfig(_ config: DensityConfig) {
            self.densityConfig = config
            // 如果有現有的識別結果，重新進行密度分析
            if !recognizedTexts.isEmpty {
                densestRegion = nil // 清除現有的密度區域
                // 重新分析密度
                DispatchQueue.main.async {
                    if let processedImage = (self.previewImage as? Image)?.asUIImage() {
                        self.analyzeDensity(from: self.recognizedTexts.map { VNRecognizedTextObservation(text: $0) }, imageSize: processedImage.size)
                    }
                }
            }
        }
        
        private func analyzeDensity(from observations: [VNRecognizedTextObservation], imageSize: CGSize) {
            // 使用配置的區域數量
            let sections = densityConfig.sections
            var densities = Array(repeating: 0, count: sections)
            
            // 計算每個區域中的文本框數量
            for observation in observations {
                let normalizedY = observation.boundingBox.midY
                let sectionIndex = Int(normalizedY * CGFloat(sections))
                if sectionIndex < sections {
                    densities[sectionIndex] += 1
                }
            }
            
            // 找出密度最高的區域
            if let maxDensityIndex = densities.indices.max(by: { densities[$0] < densities[$1] }) {
                let sectionHeight = 1.0 / CGFloat(sections)
                let y = CGFloat(maxDensityIndex) * sectionHeight
                
                // 設置密度最高區域的框
                self.densestRegion = CGRect(x: 0,
                                          y: y,
                                          width: 1.0,
                                          height: sectionHeight)
                
                logger.info("找到密度最高區域：第 \(maxDensityIndex + 1) 區，包含 \(densities[maxDensityIndex]) 個文本框")
            }
        }
        
        func processPickerResult(_ item: PhotosPickerItem?) async {
            guard let item = item else {
                logger.error("未選擇圖片")
                return
            }
            
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    logger.error("無法載入圖片數據")
                    return
                }
                
                guard let uiImage = UIImage(data: data) else {
                    logger.error("無法創建 UIImage")
                    return
                }
                
                // 處理圖片方向
                let processedImage = uiImage.imageOriented
                self.previewImage = Image(uiImage: processedImage)
                logger.info("成功載入並顯示圖片")
                
                // 創建並配置文字識別請求
                let request = VNRecognizeTextRequest { [weak self] request, error in
                    guard let self = self else { return }
                    
                    if let error = error {
                        logger.error("文字識別請求失敗: \(error.localizedDescription)")
                        return
                    }
                    
                    guard let results = request.results as? [VNRecognizedTextObservation] else {
                        logger.error("無法獲取文字識別結果")
                        return
                    }
                    
                    logger.info("檢測到 \(results.count) 個文字區域")
                    
                    Task { @MainActor in
                        self.recognizedTexts = results.compactMap { observation -> RecognizedText? in
                            guard let text = observation.topCandidates(1).first?.string else { return nil }
                            logger.info("識別到文字: \(text)")
                            return RecognizedText(text: text, boundingBox: observation.boundingBox)
                        }
                        
                        if self.recognizedTexts.isEmpty {
                            logger.error("未檢測到任何文字")
                        } else {
                            // 分析文本密度
                            self.analyzeDensity(from: results, imageSize: processedImage.size)
                            logger.info("已完成文本密度分析")
                        }
                    }
                }
                
                // 配置文字識別請求參數
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = true
                request.minimumTextHeight = 0.02
                request.recognitionLanguages = ["zh-Hant", "zh-Hans", "en-US"] // 支援繁體中文、簡體中文和英文
                
                // 使用原始圖片方向
                let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
                guard let cgImage = uiImage.cgImage else {
                    logger.error("無法獲取 CGImage")
                    return
                }
                
                let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation)
                
                // 執行文字識別請求
                try handler.perform([request])
                logger.info("已執行文字識別請求")
                
            } catch {
                logger.error("圖片處理過程發生錯誤: \(error.localizedDescription)")
            }
        }
    }
}

struct RecognizedText: Identifiable {
    var id: UUID = UUID()
    var text: String
    var boundingBox: CGRect
}

extension CGImagePropertyOrientation {
    init(_ uiOrientation: UIImage.Orientation) {
        switch uiOrientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}

extension UIImage {
    var imageOriented: UIImage {
        if imageOrientation == .up { return self }
        
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? self
    }
}

fileprivate let logger = Logger(subsystem: "com.jinshub.yolov8swiftui", category: "ContentView")
