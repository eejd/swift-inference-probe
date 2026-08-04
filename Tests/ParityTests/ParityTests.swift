import CoreML
import CryptoKit
import Foundation
import Testing
@testable import InferenceProbe

/// Core ML against the PyTorch fp32 reference, on committed inputs.
///
/// This is the one suite that genuinely needs a golden. Everything else here
/// is differential — the runtime is its own oracle — but "did the conversion
/// preserve the model?" has to be answered against something outside Core ML,
/// and that something is a tensor PyTorch produced and we committed.
///
/// The inputs are read, never regenerated. F-015 is the reason: Swift's seeded
/// generator is not `torch.rand`, so a regenerated "same" input is a different
/// tensor and any elementwise comparison against a PyTorch reference is
/// meaningless.
@Suite("Parity against the PyTorch reference",
       .enabled(if: Fixtures.modelsAvailable && Fixtures.goldensAvailable,
                "run: cd python && python export_model.py && python generate_goldens.py"))
struct ParityTests {

    let store = FileGoldenStore()

    /// The fp32 path is the **control**. It is what makes the fp16 number
    /// attributable: if conversion were lossy in itself, both numbers would be
    /// large and the fp16 figure would say nothing about precision.
    static let fp32Policy = TolerancePolicy(
        criteria: [.l1Mean(max: 1e-5),
                   .absolute(atol: 1e-4),
                   .topKAgreement(k: 1, minOverlap: 1.0)],
        rationale: """
            F-006 measured Core ML fp32 vs PyTorch fp32 at L1 1.57e-6 and max \
            abs 9.78e-6, and F-009 held it at 3.54e-6 across 200 inputs from \
            two distributions. The gates sit roughly 3x and 10x above those. \
            Top-1 is asserted here because the fp32 path has never disagreed — \
            the conversion is faithful and the residual is float noise.
            """)

    /// fp16 is the default coremltools target and the reason this project
    /// exists (SPEC §2.4). Divergence here is expected behaviour, so the
    /// assertion is on how much.
    static let fp16Policy = TolerancePolicy(
        criteria: [.l1Mean(max: 0.030),
                   .cosineSimilarity(min: 0.9990)],
        rationale: """
            F-009, n=100 per distribution: Core ML fp16 vs PyTorch fp32 L1 mean \
            0.0117 on white noise and 0.0132 on natural 1/f, max 0.0203. The \
            gate sits ~50% above the observed max because these are only four \
            committed inputs and the population spread is known to be wider.

            Top-1 is deliberately NOT asserted. F-009 measured 4/100 and 2/100 \
            top-1 flips against the PyTorch reference at fp16, so a top-1 gate \
            here would fail on 2-4% of inputs by construction — a flaky test \
            that teaches people to re-run rather than a gate that means \
            anything (TEST_STRATEGY §7). Top-1 stability is asserted where it \
            actually holds: across compute units, in ComputeUnitTests.
            """)

    func predict(_ golden: Golden,
                 _ precision: ModelRunner.Precision) async throws -> [Double] {
        let runner = try await ModelRunner(modelURL: Fixtures.modelURL(precision),
                                           computeUnits: .cpuOnly)
        return try runner.predict(golden.input,
                                  inputName: Fixtures.inputName,
                                  outputName: Fixtures.outputName)
    }

    @Test("every committed golden loads and verifies its digest")
    func goldensLoad() throws {
        let ids = store.availableIDs
        #expect(!ids.isEmpty, "no goldens — run python/generate_goldens.py")
        for id in ids {
            // `load` throws on a digest or size mismatch, so this is the
            // fixture-integrity check as well as a smoke test.
            let golden = try store.load(id)
            #expect(golden.input.count == 3 * 224 * 224)
            #expect(golden.reference.count == Fixtures.classCount)
            #expect(!golden.manifest.reason.isEmpty)
        }
    }

    @Test("Core ML fp32 matches the PyTorch reference")
    func fp32Parity() async throws {
        for id in store.availableIDs {
            let golden = try store.load(id)
            let actual = try await predict(golden, .fp32)
            let result = TensorComparison.compare(
                actual, to: golden.reference,
                using: Self.fp32Policy, shape: golden.manifest.reference.shape)

            #expect(result.passed, """
                fp32 parity failed for \(id).
                \(result.failureDescription)
                This is the control path — if it fails, the conversion itself is \
                suspect and no fp16 number below is interpretable.
                """)
        }
    }

    @Test("Core ML fp16 drift stays within the measured envelope")
    func fp16Parity() async throws {
        for id in store.availableIDs {
            let golden = try store.load(id)
            let actual = try await predict(golden, .fp16)
            let result = TensorComparison.compare(
                actual, to: golden.reference,
                using: Self.fp16Policy, shape: golden.manifest.reference.shape)

            #expect(result.passed, """
                fp16 drift for \(id) exceeded the F-009 envelope.
                \(result.failureDescription)
                Localize with: cd python && python localize.py --arm precision
                """)
        }
    }

    /// The control has to be *demonstrably* tighter than the thing it controls
    /// for, or it is not doing its job. F-006 measured a ~7000x gap; asserting
    /// two orders of magnitude leaves ample room while still failing loudly if
    /// the fp32 path ever starts drifting like the fp16 one.
    @Test("the fp32 control is orders of magnitude tighter than fp16")
    func controlIsTighterThanSubject() async throws {
        let id = try #require(store.availableIDs.first)
        let golden = try store.load(id)

        let fp32Drift = TensorComparison.measure(
            .l1Mean(max: 0), actual: try await predict(golden, .fp32),
            reference: golden.reference)
        let fp16Drift = TensorComparison.measure(
            .l1Mean(max: 0), actual: try await predict(golden, .fp16),
            reference: golden.reference)

        #expect(fp32Drift * 100 < fp16Drift, """
            The fp32 control (L1 \(fp32Drift)) is not meaningfully tighter than \
            the fp16 path (L1 \(fp16Drift)). The control exists to attribute \
            fp16 drift to precision rather than to a broken conversion; if the \
            two are comparable it cannot do that.
            """)
    }

    /// Goldens are only valid for the model they were blessed against. F-011
    /// established that the identity must be the **weights** — the `.mlpackage`
    /// carries fresh UUIDs and a date stamp on every export, so a package
    /// digest would report a false mismatch after any re-export.
    @Test("goldens were blessed against the models currently on disk")
    func goldensMatchCurrentModel() throws {
        for id in store.availableIDs {
            let golden = try store.load(id)
            guard let recorded = golden.manifest.modelWeightSha256 else { continue }
            for (precision, expected) in recorded {
                guard let precisionCase = ModelRunner.Precision(rawValue: precision) else { continue }
                let weights = Fixtures.modelURL(precisionCase)
                    .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
                let data = try Data(contentsOf: weights)
                let actual = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()

                #expect(actual == expected, """
                    \(id) was blessed against a different \(precision) model.
                      manifest \(expected.prefix(16))…
                      on disk  \(actual.prefix(16))…
                    Re-export changed the weights, or the goldens are stale. \
                    Re-bless deliberately with python/generate_goldens.py and a \
                    stated reason — do not widen a tolerance to absorb this.
                    """)
            }
        }
    }
}
