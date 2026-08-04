import CoreML
import Foundation

/// Locates the committed fixtures and regenerates the inputs the Python
/// experiments use.
///
/// The `.mlpackage` files are **gitignored** — large, and byte-reproducible
/// from `export_model.py` (F-011). So they are present on a development machine
/// and absent on a fresh clone or a cold CI runner, and every suite that needs
/// one has to degrade to a skip with an actionable message rather than a crash.
public enum Fixtures {

    public static let inputName = "input"
    public static let outputName = "var_930"
    public static let inputShape = [1, 3, 224, 224]
    public static let classCount = 1000

    /// Walks up from this source file to the package root. The models are not
    /// SwiftPM resources — they are large build artifacts that happen to live
    /// in the repository — so `Bundle.module` does not see them.
    public static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)          // Sources/InferenceProbe/Fixtures.swift
            .deletingLastPathComponent()          // Sources/InferenceProbe
            .deletingLastPathComponent()          // Sources
            .deletingLastPathComponent()          // <root>
    }

    public static var modelsDirectory: URL {
        packageRoot.appendingPathComponent("Fixtures/models")
    }

    public static func modelURL(_ precision: ModelRunner.Precision) -> URL {
        modelsDirectory.appendingPathComponent(
            "mobilenet_v3_large.\(precision.rawValue).mlpackage")
    }

    public static func modelExists(_ precision: ModelRunner.Precision) -> Bool {
        FileManager.default.fileExists(atPath: modelURL(precision).path)
    }

    /// True when every model a suite might need is present.
    public static var modelsAvailable: Bool {
        ModelRunner.Precision.allCases.allSatisfy(modelExists)
    }

    // MARK: - Inputs

    /// ImageNet channel statistics, matching `export_model.py`.
    static let mean: [Double] = [0.485, 0.456, 0.406]
    static let standardDeviation: [Double] = [0.229, 0.224, 0.225]

    /// A deterministic ImageNet-scaled input.
    ///
    /// This is **not** bit-identical to the Python fixture: `torch.rand` with a
    /// given seed is a specific generator that Swift does not reproduce. So
    /// Swift-side numbers are comparable to each other, and to Python only in
    /// distribution and magnitude — never elementwise. Where an exact match
    /// matters, the tensor has to be committed to `Fixtures/inputs/` and read,
    /// not regenerated.
    public static func normalizedInput(seed: UInt64 = 0) -> [Double] {
        var generator = SplitMix64(seed: seed)
        let pixelsPerChannel = inputShape[2] * inputShape[3]
        var values = [Double]()
        values.reserveCapacity(3 * pixelsPerChannel)
        for channel in 0..<3 {
            for _ in 0..<pixelsPerChannel {
                let uniform = generator.nextUnitDouble()
                values.append((uniform - mean[channel]) / standardDeviation[channel])
            }
        }
        return values
    }

    /// A filled input of one repeated value — the malformed-input battery from
    /// F-013, which is where Core ML and PyTorch stopped agreeing.
    public static func filledInput(_ value: Double) -> [Double] {
        [Double](repeating: value, count: 3 * inputShape[2] * inputShape[3])
    }

    /// A deterministic generator, so a "random" input is reproducible without
    /// depending on the platform's RNG.
    struct SplitMix64 {
        private var state: UInt64
        init(seed: UInt64) { state = seed }

        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }

        mutating func nextUnitDouble() -> Double {
            Double(next() >> 11) * (1.0 / 9007199254740992.0)   // 2^-53
        }
    }
}
