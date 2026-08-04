import Testing
@testable import InferenceProbe

/// `TensorComparison` is the arithmetic every assertion in the project rests
/// on, and it has to agree with `drift()` in `python/export_model.py` — the
/// Swift side asserts, the Python side produces references, and numbers from
/// the two are compared directly in FINDINGS. So the formulas are pinned here
/// against hand-computable cases rather than against the implementation.
@Suite("TensorComparison")
struct TensorComparisonTests {

    let environment = HostEnvironment(chip: .m4Pro,
                                      chipBrandString: "Apple M4 Pro",
                                      osVersion: "26.5.2")

    @Test("identical tensors satisfy every criterion")
    func identicalTensorsPass() {
        let values = [1.0, -2.0, 3.5, 0.0]
        let policy = TolerancePolicy(
            criteria: [.absolute(atol: 0), .l1Mean(max: 0),
                       .cosineSimilarity(min: 1.0),
                       .topKAgreement(k: 2, minOverlap: 1.0),
                       .klDivergence(maxNats: 0)],
            rationale: "identity: every criterion must be exactly satisfied")

        let result = TensorComparison.compare(values, to: values,
                                              using: policy, environment: environment)
        #expect(result.passed, "\(result.failureDescription)")
        #expect(result.worstDeviation == 0)
    }

    @Test("l1Mean is the mean absolute difference")
    func l1Mean() {
        // deviations 1, 2, 3, 6 → mean 3
        let measured = TensorComparison.measure(
            .l1Mean(max: 0), actual: [1, 2, 3, 6], reference: [0, 0, 0, 0])
        #expect(abs(measured - 3.0) < 1e-12)
    }

    @Test("absolute reports the largest deviation, not the mean")
    func absoluteIsMax() {
        let measured = TensorComparison.measure(
            .absolute(atol: 0), actual: [1, 2, 3, 6], reference: [0, 0, 0, 0])
        #expect(measured == 6)
    }

    @Test("cosine similarity is 1 for parallel and 0 for orthogonal vectors")
    func cosine() {
        let parallel = TensorComparison.measure(
            .cosineSimilarity(min: 0), actual: [2, 4], reference: [1, 2])
        #expect(abs(parallel - 1.0) < 1e-12)

        let orthogonal = TensorComparison.measure(
            .cosineSimilarity(min: 0), actual: [1, 0], reference: [0, 1])
        #expect(abs(orthogonal) < 1e-12)
    }

    /// The F-008 case, in miniature: a scaled tensor is perfectly aligned by
    /// cosine while every element has moved. This is why the compute-unit
    /// policy pairs cosine with l1Mean instead of trusting cosine alone.
    @Test("cosine is blind to a uniform rescale that l1Mean catches")
    func cosineIsBlindToScale() {
        let reference = [1.0, 2.0, 3.0]
        let scaled = reference.map { $0 * 10 }

        let cosine = TensorComparison.measure(
            .cosineSimilarity(min: 0), actual: scaled, reference: reference)
        let l1 = TensorComparison.measure(
            .l1Mean(max: 0), actual: scaled, reference: reference)

        #expect(abs(cosine - 1.0) < 1e-12, "cosine should see no difference")
        #expect(l1 > 1.0, "l1Mean should see a large one")
    }

    @Test("topKAgreement counts overlap of the k largest indices")
    func topK() {
        // top-2 of a is {3,2}; of b is {3,1} → overlap 1/2
        let measured = TensorComparison.measure(
            .topKAgreement(k: 2, minOverlap: 0),
            actual: [0, 5, 9, 10], reference: [0, 9, 5, 10])
        #expect(abs(measured - 0.5) < 1e-12)
    }

    @Test("mixed reports the excess beyond the relative allowance")
    func mixed() {
        // |a-b| = 0.5, rtol*|b| = 0.1*10 = 1.0 → excess -0.5, within atol
        let measured = TensorComparison.measure(
            .mixed(atol: 0.1, rtol: 0.1), actual: [10.5], reference: [10.0])
        #expect(abs(measured - (-0.5)) < 1e-12)
    }

    @Test("relative treats a zero reference with a non-zero actual as infinite")
    func relativeAgainstZero() {
        let diverged = TensorComparison.measure(
            .relative(rtol: 0), actual: [1.0], reference: [0.0])
        #expect(diverged.isInfinite, "a non-zero value against zero has no finite ratio")

        let identical = TensorComparison.measure(
            .relative(rtol: 0), actual: [0.0], reference: [0.0])
        #expect(identical == 0)
    }

    @Test("the worst element is reported as a tensor coordinate")
    func worstIndexIsACoordinate() {
        var actual = [Double](repeating: 0, count: 6)
        actual[4] = 9                                   // row 1, column 1 of 2x3
        let policy = TolerancePolicy(criteria: [.absolute(atol: 100)],
                                     rationale: "loose: this test is about the coordinate")

        let result = TensorComparison.compare(
            actual, to: [Double](repeating: 0, count: 6),
            using: policy, environment: environment, shape: [2, 3])

        #expect(result.worstIndex == [1, 1])
        #expect(result.worstDeviation == 9)
    }

    @Test("a failure names the criterion, the numbers, and the host chip")
    func failureDescriptionIsActionable() {
        let policy = TolerancePolicy(criteria: [.l1Mean(max: 0.01)],
                                     rationale: "tight: this test expects a failure")
        let result = TensorComparison.compare([1.0], to: [0.0],
                                              using: policy, environment: environment)

        #expect(!result.passed)
        let message = result.failureDescription
        #expect(message.contains("l1Mean"))
        #expect(message.contains("over limit"))
        #expect(message.contains("Apple M4 Pro"), "the chip is load-bearing, not decoration")
    }

    @Test("chip overrides replace the base criteria for that chip only")
    func chipOverrides() {
        let policy = TolerancePolicy(
            criteria: [.l1Mean(max: 0.001)],
            rationale: "base is strict; M1 is known to diverge more (F-010)",
            chipOverrides: [.m1: [.l1Mean(max: 1.0)]])

        let m1 = HostEnvironment(chip: .m1, chipBrandString: "Apple M1", osVersion: "26.1")
        #expect(policy.criteria(for: environment) == [.l1Mean(max: 0.001)])
        #expect(policy.criteria(for: m1) == [.l1Mean(max: 1.0)])

        // Same tensors, opposite verdicts, purely because of the host chip.
        #expect(!TensorComparison.compare([0.5], to: [0.0], using: policy,
                                          environment: environment).passed)
        #expect(TensorComparison.compare([0.5], to: [0.0], using: policy,
                                         environment: m1).passed)
    }
}
