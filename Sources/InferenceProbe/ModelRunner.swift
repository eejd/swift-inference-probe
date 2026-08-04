import CoreML
import Foundation

/// A Core ML model pinned to one compute unit.
///
/// The compute unit is fixed at construction rather than per-prediction,
/// because the whole differential design depends on knowing which path
/// produced a number. F-014 is the reason to be careful about the distinction:
/// `.cpuAndNE` compiles to a pure-CPU plan for an fp32 model, so "I asked for
/// the Neural Engine" and "it ran on the Neural Engine" are different claims.
/// **Not `Sendable`, deliberately.** SPEC §4.4 declared it as such, but
/// `MLModel` does not conform under Swift 6 strict concurrency, and marking the
/// wrapper `@unchecked Sendable` would be asserting thread safety this project
/// has not measured. TEST_STRATEGY §5.1 lists concurrent `prediction()` as a
/// Tier 1 test precisely because it is not yet established; when that test
/// exists and passes, the conformance can be added with evidence behind it.
public struct ModelRunner {

    public enum Precision: String, Sendable, CaseIterable { case fp32, fp16 }

    public enum Failure: Error, CustomStringConvertible {
        case modelNotFound(URL)
        case noInputDescription(String)
        case unexpectedOutput(String)

        public var description: String {
            switch self {
            case .modelNotFound(let url):
                return """
                no model at \(url.path)
                .mlpackage files are gitignored (large, and reproducible — F-011).
                Regenerate with:  cd python && conda activate inferenceprobe && python export_model.py
                """
            case .noInputDescription(let name):
                return "model has no input named '\(name)'"
            case .unexpectedOutput(let name):
                return "model produced no MLMultiArray for output '\(name)'"
            }
        }
    }

    public let model: MLModel
    public let computeUnits: MLComputeUnits
    public let environment: HostEnvironment

    /// Compiles the `.mlpackage` and loads it against `computeUnits`.
    ///
    /// Compilation is not cached across instances on purpose: a runner is cheap
    /// and the cost belongs to whoever is measuring load time.
    public init(modelURL: URL, computeUnits: MLComputeUnits) async throws {
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw Failure.modelNotFound(modelURL)
        }
        let compiledURL = try await MLModel.compileModel(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.computeUnits = computeUnits
        self.environment = .current()
    }

    /// Runs one prediction and returns the named output flattened to `[Double]`.
    ///
    /// `[Double]` rather than `MLMultiArray` because every comparison in
    /// `TensorComparison` is in double precision, and doing the widening once
    /// here keeps the fp16/fp32 distinction out of the comparison code.
    public func predict(_ values: [Double],
                        inputName: String,
                        outputName: String) throws -> [Double] {
        let array = try makeInputArray(values, named: inputName)
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: array])
        let output = try model.prediction(from: provider)
        guard let result = output.featureValue(for: outputName)?.multiArrayValue else {
            throw Failure.unexpectedOutput(outputName)
        }
        return (0..<result.count).map { result[$0].doubleValue }
    }

    /// Builds the input in whatever type the model actually declares.
    ///
    /// The fp16 export declares a **float16** input, so handing it a float32
    /// array is not guaranteed to work. Reading the type off the model rather
    /// than assuming keeps one runner correct for both precisions.
    private func makeInputArray(_ values: [Double], named inputName: String) throws -> MLMultiArray {
        guard let constraint = model.modelDescription
            .inputDescriptionsByName[inputName]?.multiArrayConstraint else {
            throw Failure.noInputDescription(inputName)
        }
        let array = try MLMultiArray(shape: constraint.shape,
                                     dataType: constraint.dataType)
        precondition(array.count == values.count,
                     "input expects \(array.count) values, got \(values.count)")
        for i in values.indices {
            array[i] = NSNumber(value: values[i])
        }
        return array
    }

    /// Cold = compile from `.mlpackage`; warm = load an already-compiled
    /// `.mlmodelc`. These are commonly conflated and differ by orders of
    /// magnitude, so they are separate measurements (SPEC §4.4).
    public static func measureLoad(modelURL: URL,
                                   computeUnits: MLComputeUnits,
                                   cold: Bool) async throws -> Duration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        if cold {
            let start = ContinuousClock.now
            let compiledURL = try await MLModel.compileModel(at: modelURL)
            _ = try MLModel(contentsOf: compiledURL, configuration: configuration)
            return ContinuousClock.now - start
        }
        // Warm: pay the compile outside the measured window.
        let compiledURL = try await MLModel.compileModel(at: modelURL)
        let start = ContinuousClock.now
        _ = try MLModel(contentsOf: compiledURL, configuration: configuration)
        return ContinuousClock.now - start
    }
}
