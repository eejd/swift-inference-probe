import CoreML
import Testing
@testable import InferenceProbe

/// Differential testing across compute units — the highest-value suite here,
/// because it needs no ground truth. The same file, the same weights, the same
/// input, four execution paths: they must agree, and disagreement is a defect
/// regardless of which path is wrong (docs/TEST_STRATEGY.md §2.1).
///
/// Every unit is compared against `.cpuOnly`, never pairwise-all: O(n) instead
/// of O(n²), and a failure has one clear referent (SPEC §2.3).
///
/// Thresholds come from F-008's n=200 measurement, not from round numbers.
/// The noise floor there was exactly zero — every unit bit-identical across 20
/// repeats — so these gates need headroom only for variation across inputs, not
/// for jitter, and they cannot flake for numerical reasons.
@Suite("Compute-unit differential",
       .enabled(if: Fixtures.modelsAvailable,
                "no .mlpackage — run: cd python && python export_model.py"))
struct ComputeUnitTests {

    static let anePolicy = TolerancePolicy(
        criteria: [.l1Mean(max: 0.030), .cosineSimilarity(min: 0.9990)],
        rationale: """
            F-008, n=200 against the .cpuOnly anchor: ANE L1 p95 0.0218, max \
            0.0242, cosine min 0.999574. The gate sits ~25% above the observed \
            max. The noise floor is exactly 0.0 (20 repeats/unit), so no \
            allowance for run-to-run jitter is needed. L1 and cosine together \
            because cosine alone never dropped below 0.9995 while 1.5% of \
            inputs changed top-1 class — cosine is blind to the failure that \
            actually matters.
            """)

    static let gpuPolicy = TolerancePolicy(
        criteria: [.l1Mean(max: 0.020), .cosineSimilarity(min: 0.9990)],
        rationale: """
            F-008, n=200: GPU L1 p95 0.0124, max 0.0149, cosine min 0.999826. \
            Tighter than the ANE gate because the GPU diverges from CPU less, \
            consistent with the Neural Engine using the most aggressive \
            numerics of the three. Zero noise floor, as above.
            """)

    static func policy(for units: MLComputeUnits) -> TolerancePolicy {
        units == .cpuAndGPU ? gpuPolicy : anePolicy
    }

    /// The anchor's own output, computed once per test.
    static func anchorOutput(_ input: [Double]) async throws -> [Double] {
        let runner = try await ModelRunner(modelURL: Fixtures.modelURL(.fp16),
                                           computeUnits: .cpuOnly)
        return try runner.predict(input, inputName: Fixtures.inputName,
                                  outputName: Fixtures.outputName)
    }

    @Test("each compute unit agrees with the .cpuOnly anchor",
          arguments: [MLComputeUnits.cpuAndGPU, .cpuAndNeuralEngine, .all])
    func agreesWithAnchor(units: MLComputeUnits) async throws {
        let input = Fixtures.normalizedInput()
        let anchor = try await Self.anchorOutput(input)

        let runner = try await ModelRunner(modelURL: Fixtures.modelURL(.fp16),
                                           computeUnits: units)
        let actual = try runner.predict(input, inputName: Fixtures.inputName,
                                        outputName: Fixtures.outputName)

        let result = TensorComparison.compare(
            actual, to: anchor,
            using: Self.policy(for: units),
            shape: [1, Fixtures.classCount])

        #expect(result.passed, """
            \(describe(units)) disagrees with .cpuOnly beyond tolerance.
            \(result.failureDescription)
            policy rationale: \(Self.policy(for: units).rationale)
            """)
    }

    /// F-014 read the compute plans directly and found `.all` and `.cpuAndNE`
    /// identical for this fp16 model — same device, same 182 ops. So this is
    /// not two paths that happen to agree; it is one plan reached two ways.
    ///
    /// The equality is therefore the assertion: if it ever breaks, Core ML's
    /// dispatch has changed underneath us, and nothing else in the suite would
    /// report it (F-007, TEST_STRATEGY §5.3).
    @Test("`.all` is bit-identical to `.cpuAndNeuralEngine` for this fp16 model")
    func allMatchesNeuralEngine() async throws {
        let input = Fixtures.normalizedInput()

        let ne = try await ModelRunner(modelURL: Fixtures.modelURL(.fp16),
                                       computeUnits: .cpuAndNeuralEngine)
        let all = try await ModelRunner(modelURL: Fixtures.modelURL(.fp16),
                                        computeUnits: .all)
        let neOutput = try ne.predict(input, inputName: Fixtures.inputName,
                                      outputName: Fixtures.outputName)
        let allOutput = try all.predict(input, inputName: Fixtures.inputName,
                                        outputName: Fixtures.outputName)

        #expect(neOutput == allOutput, """
            .all diverged from .cpuAndNeuralEngine. F-014 measured identical \
            compute plans for these two on the fp16 model, so this is a \
            dispatch change rather than a numerical one — re-run \
            python/compute_plan.py and compare against \
            Fixtures/models/compute_plan.json.
            """)
    }

    /// F-008 measured a zero noise floor: same input, same unit, 20 repeats,
    /// bit-identical every time. That is what licenses the tight thresholds
    /// above, so it is asserted rather than assumed — if it ever stops holding,
    /// every threshold in this suite needs revisiting.
    @Test("a single compute unit is bit-reproducible across repeats",
          arguments: [MLComputeUnits.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine])
    func noiseFloorIsZero(units: MLComputeUnits) async throws {
        let input = Fixtures.normalizedInput()
        let runner = try await ModelRunner(modelURL: Fixtures.modelURL(.fp16),
                                           computeUnits: units)

        let first = try runner.predict(input, inputName: Fixtures.inputName,
                                       outputName: Fixtures.outputName)
        for repeatIndex in 1..<5 {
            let again = try runner.predict(input, inputName: Fixtures.inputName,
                                           outputName: Fixtures.outputName)
            #expect(again == first, """
                \(describe(units)) is not bit-reproducible: repeat \
                \(repeatIndex) differs from the first prediction. F-008 \
                measured a zero noise floor across 20 repeats; if that no \
                longer holds, the thresholds in this suite are no longer \
                justified.
                """)
        }
    }

    func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly:            return ".cpuOnly"
        case .cpuAndGPU:          return ".cpuAndGPU"
        case .cpuAndNeuralEngine: return ".cpuAndNeuralEngine"
        case .all:                return ".all"
        @unknown default:         return "unknown compute unit"
        }
    }
}
