//
//  ContentView.swift
//  YOLOv8SwiftUI
//
//  Created by Jin on 2024-03-19.
//

import CoreML
import os.log
import PhotosUI
import SwiftUI
import UIKit
import Vision

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
        let cgRect = denormalize(imageViewSize: imageViewGeometry.size, normalizedCGRect: rect)
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

    @MainActor class DataModel: ObservableObject {
        @Published var previewImage: Image?
        @Published var recognizedTexts: [RecognizedText] = []
        @Published var densestRegion: CGRect?

        // 密度分析配置
        struct DensityConfig {
            // 基本設置
            var sections: Int           // 垂直區域數量（劃分成幾個垂直條帶）
            var minimumDensity: Int    // 最小文字數量閾值
            
            // 進階設置
            var windowSize: Int        // 滑動視窗大小（用於平滑處理）
            var overlap: Double        // 重疊比例 (0.0 - 1.0)
            var densityMethod: DensityMethod  // 密度計算方法
            
            // 密度計算方法
            enum DensityMethod {
                case count      // 純文字框數量
                case area      // 文字框面積
                case weighted  // 加權（考慮文字長度）
            }
            
            static let `default` = DensityConfig(
                sections: 10,          // 較細的垂直分區
                minimumDensity: 3,      // 最少需要3個文字框
                windowSize: 5,          // 5個區域的滑動視窗
                overlap: 0.5,           // 50%的重疊
                densityMethod: .weighted // 預設使用加權方法
            )
        }

        var densityConfig: DensityConfig = .default

        init(densityConfig: DensityConfig = .default) {
            self.densityConfig = densityConfig
        }

        // 更新密度分析配置
        func updateDensityConfig(_ config: DensityConfig) {
            densityConfig = config
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
            let sections = densityConfig.sections
            var densities = Array(repeating: 0.0, count: sections)
            
            // 根據選擇的方法計算密度
            for observation in observations {
                let normalizedX = observation.boundingBox.midX
                let sectionIndex = Int(normalizedX * CGFloat(sections))
                guard sectionIndex < sections else { continue }
                
                switch densityConfig.densityMethod {
                case .count:
                    densities[sectionIndex] += 1
                case .area:
                    densities[sectionIndex] += Double(observation.boundingBox.width * observation.boundingBox.height)
                case .weighted:
                    let candidates: [VNRecognizedText] = observation.topCandidates(1)
                    if let text = candidates.first?.string {
                        densities[sectionIndex] += Double(text.count) // 文字長度權重
                    }
                }
            }
            
            // 使用滑動視窗進行平滑處理
            var smoothedDensities = Array(repeating: 0.0, count: sections)
            let halfWindow = densityConfig.windowSize / 2
            
            for i in 0..<sections {
                var sum = 0.0
                var count = 0
                
                // 計算重疊視窗內的平均密度
                let windowStart = max(0, i - halfWindow)
                let windowEnd = min(sections - 1, i + halfWindow)
                for j in windowStart...windowEnd {
                    sum += densities[j]
                    count += 1
                }
                
                smoothedDensities[i] = sum / Double(count)
            }
            
            // 找出密度最高的區域
            if let maxDensityIndex = smoothedDensities.indices.max(by: { smoothedDensities[$0] < smoothedDensities[$1] }),
               smoothedDensities[maxDensityIndex] >= Double(densityConfig.minimumDensity) {
                
                // 考慮重疊計算最終區域
                let sectionWidth = 1.0 / CGFloat(sections)
                let baseX = CGFloat(maxDensityIndex) * sectionWidth
                let expandedWidth = sectionWidth * (1.0 + CGFloat(densityConfig.overlap))
                
                // 設置密度最高區域的框（垂直條帶）
                densestRegion = CGRect(x: baseX - (expandedWidth - sectionWidth) / 2,
                                     y: 0,
                                     width: expandedWidth,
                                     height: 1.0)
                
                // 輸出詳細資訊
                print("""
                    找到密度最高區域：
                    - 區域索引：第 \(maxDensityIndex + 1) 區
                    - 原始密度：\(densities[maxDensityIndex])
                    - 平滑後密度：\(smoothedDensities[maxDensityIndex])
                    - 計算方法：\(densityConfig.densityMethod)
                    - 視窗大小：\(densityConfig.windowSize)
                    - 重疊比例：\(densityConfig.overlap * 100)%
                    """)
            } else {
                densestRegion = nil
                logger.info("沒有找到符合最小密度要求的區域")
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
                previewImage = Image(uiImage: processedImage)
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

extension Image {
    func asUIImage() -> UIImage? {
        let controller = UIHostingController(rootView: self)
        let view = controller.view

        let targetSize = controller.view.intrinsicContentSize
        view?.bounds = CGRect(origin: .zero, size: targetSize)
        view?.backgroundColor = .clear

        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            view?.drawHierarchy(in: controller.view.bounds, afterScreenUpdates: true)
        }
    }
}

extension VNRecognizedTextObservation {
    convenience init(text: RecognizedText) {
        self.init()
        setValue(text.boundingBox, forKey: "boundingBox")
    }
}

fileprivate let logger = Logger(subsystem: "com.jinshub.yolov8swiftui", category: "ContentView")
