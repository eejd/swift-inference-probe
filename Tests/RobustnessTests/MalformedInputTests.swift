import CoreML
import Testing
@testable import InferenceProbe

/// What happens when the input is wrong.
///
/// F-013 found that Core ML, driven from Python, returns a confident finite
/// answer for all-NaN and all-Inf input where PyTorch correctly propagates NaN.
/// That finding could not say **where** the sanitization happens — the
/// coremltools Python wrapper, the framework, or a backend compiler — and named
/// driving the same inputs through Swift as the experiment that separates them.
///
/// This suite is that experiment, plus the robustness assertions that follow
/// from the answer.
@Suite("Malformed input",
       .enabled(if: Fixtures.modelsAvailable,
                "no .mlpackage — run: cd python && python export_model.py"))
struct MalformedInputTests {

    static func predict(_ input: [Double],
                        _ precision: ModelRunner.Precision = .fp16,
                        _ units: MLComputeUnits = .cpuOnly) async throws -> [Double] {
        let runner = try await ModelRunner(modelURL: Fixtures.modelURL(precision),
                                           computeUnits: units)
        return try runner.predict(input, inputName: Fixtures.inputName,
                                  outputName: Fixtures.outputName)
    }

    /// The floor: whatever the numerics, the runtime must not crash or hang on
    /// a well-shaped tensor full of nonsense. Tier 1, TEST_STRATEGY §5.1.
    @Test("malformed input does not crash the runtime", arguments: [
        Double.nan, .infinity, -.infinity, 0, 1e30, -1e30,
    ])
    func doesNotCrash(fillValue: Double) async throws {
        let output = try await Self.predict(Fixtures.filledInput(fillValue))
        #expect(output.count == Fixtures.classCount)
    }

    /// **The F-013 follow-up.** Same inputs, same model, Swift instead of Python.
    ///
    /// This pins observed behaviour rather than endorsing it. Swallowing NaN is
    /// a hazard: a pipeline bug that feeds garbage to this model gets a normal
    /// class prediction back rather than an obvious failure, and PyTorch would
    /// have made the same bug visible in development. The test exists so that
    /// if Apple ever changes it, we find out deliberately instead of by
    /// surprise.
    @Test("Core ML returns finite output for NaN and Inf input", arguments: [
        Double.nan, .infinity,
    ])
    func sanitizesNonFiniteInput(fillValue: Double) async throws {
        let output = try await Self.predict(Fixtures.filledInput(fillValue))
        let nonFinite = output.filter { !$0.isFinite }.count

        #expect(nonFinite == 0, """
            Core ML produced \(nonFinite) non-finite outputs for an all-\
            \(fillValue) input via Swift, where the Python path produced none \
            (F-013). If this now fails, the sanitization was in the coremltools \
            Python wrapper rather than the framework — which is the open \
            question F-013 recorded, and this failure is the answer.
            """)
    }

    /// The consequence, stated as an assertion.
    ///
    /// F-013's judgement was that the Tier 1 NaN/Inf gate is a sound sufficient
    /// condition and a weak necessary one, so the suite must not rely on it
    /// alone. The mechanism that *does* catch saturation is the differential —
    /// so check that the mechanism actually works rather than assuming it.
    @Test("the fp32/fp16 differential catches saturation the NaN gate misses")
    func differentialCatchesSaturation() async throws {
        // Nominal: the two precisions agree closely (F-006, L1 ~0.011).
        let nominal = Fixtures.normalizedInput()
        let nominal32 = try await Self.predict(nominal, .fp32)
        let nominal16 = try await Self.predict(nominal, .fp16)

        let agreement = TensorComparison.measure(
            .l1Mean(max: 0), actual: nominal16, reference: nominal32)

        // Scaled far past fp16's range: F-013 measured fp16 collapsing near
        // input scale 1e3 while fp32 tracks linearly to about 1e30.
        let extreme = nominal.map { $0 * 1e4 }
        let extreme32 = try await Self.predict(extreme, .fp32)
        let extreme16 = try await Self.predict(extreme, .fp16)

        let divergence = TensorComparison.measure(
            .l1Mean(max: 0), actual: extreme16, reference: extreme32)

        // Every value here is finite, so the NaN/Inf gate reports clean.
        // Evaluated outside `#expect` because `allSatisfy` is `rethrows` and
        // the macro expansion cannot see that this closure does not throw.
        let extreme16IsFinite = extreme16.allSatisfy(\.isFinite)
        let extreme32IsFinite = extreme32.allSatisfy(\.isFinite)
        #expect(extreme16IsFinite)
        #expect(extreme32IsFinite)

        // The differential is what notices. Demanding a large multiple rather
        // than merely "greater" keeps this from passing on ordinary drift.
        #expect(divergence > agreement * 100, """
            The fp32/fp16 differential did not separate a saturated input from \
            a nominal one: nominal L1 \(agreement), extreme L1 \(divergence). \
            TEST_STRATEGY §5.1 relies on this signal precisely because the \
            NaN/Inf gate cannot see finite-but-wrong output (F-013).
            """)
    }

    /// Zeros are a legitimate input, not a malformed one, and must produce a
    /// stable answer. Included because "all zeros" is the input a buffer bug
    /// most often produces, and it is easy to leave untested.
    @Test("an all-zero input is handled and is reproducible")
    func zeroInput() async throws {
        let zeros = Fixtures.filledInput(0)
        let first = try await Self.predict(zeros)
        let second = try await Self.predict(zeros)
        let firstIsFinite = first.allSatisfy(\.isFinite)
        #expect(firstIsFinite)
        #expect(first == second)
    }
}
