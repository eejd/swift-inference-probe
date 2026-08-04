import Foundation

/// Evaluates a `TolerancePolicy` against two tensors.
///
/// Mirrors `drift()` in `python/export_model.py` deliberately: the Swift suite
/// asserts and the Python side produces references and localizes failures, so
/// the two must compute the same statistics or their numbers cannot be
/// compared. If you change a formula here, change it there.
public enum TensorComparison {

    /// - Parameters:
    ///   - reference: the anchor. For compute-unit differentials this is
    ///     always `.cpuOnly` (SPEC §2.3); for parity it is the PyTorch output.
    ///   - shape: used only to expand `worstIndex` into a coordinate.
    public static func compare(
        _ actual: [Double],
        to reference: [Double],
        using policy: TolerancePolicy,
        environment: HostEnvironment = .current(),
        shape: [Int]? = nil,
        suspectOps: [String] = [],
        suspectOpsAtol: Double? = nil
    ) -> ComparisonResult {
        precondition(actual.count == reference.count,
                     "tensor length mismatch: \(actual.count) vs \(reference.count)")
        precondition(!actual.isEmpty, "cannot compare empty tensors")

        var worstDeviation = 0.0
        var worstFlatIndex = 0
        for i in actual.indices {
            let deviation = abs(actual[i] - reference[i])
            if deviation > worstDeviation {
                worstDeviation = deviation
                worstFlatIndex = i
            }
        }

        let outcomes = policy.criteria(for: environment).map { criterion in
            let measured = measure(criterion, actual: actual, reference: reference)
            let threshold = criterion.threshold
            let passed = criterion.higherIsBetter ? measured >= threshold : measured <= threshold
            return ComparisonResult.CriterionOutcome(
                criterion: criterion, measured: measured,
                threshold: threshold, passed: passed)
        }

        return ComparisonResult(
            outcomes: outcomes,
            environment: environment,
            worstIndex: shape.map { expand(worstFlatIndex, shape: $0) } ?? [worstFlatIndex],
            worstDeviation: worstDeviation,
            suspectOps: suspectOps,
            suspectOpsAtol: suspectOpsAtol)
    }

    static func measure(_ criterion: TolerancePolicy.Criterion,
                        actual: [Double], reference: [Double]) -> Double {
        switch criterion {
        case .absolute:
            return zip(actual, reference).map { abs($0 - $1) }.max() ?? 0

        case .relative:
            // Guard the divisor rather than skipping the element: a reference
            // of exactly zero with a non-zero actual is a real failure, and
            // dropping it would hide the worst case.
            return zip(actual, reference).map { a, b in
                abs(b) > 0 ? abs(a - b) / abs(b) : (a == b ? 0 : .infinity)
            }.max() ?? 0

        case .mixed(_, let rtol):
            // Reported as the excess beyond the relative allowance, so the
            // measured value is directly comparable to `atol`.
            return zip(actual, reference).map { a, b in
                abs(a - b) - rtol * abs(b)
            }.max() ?? 0

        case .cosineSimilarity:
            let dot = zip(actual, reference).reduce(0) { $0 + $1.0 * $1.1 }
            let na = sqrt(actual.reduce(0) { $0 + $1 * $1 })
            let nb = sqrt(reference.reduce(0) { $0 + $1 * $1 })
            return (na > 0 && nb > 0) ? dot / (na * nb) : (na == nb ? 1 : 0)

        case .topKAgreement(let k, _):
            let ka = Set(topIndices(actual, k: k))
            let kb = Set(topIndices(reference, k: k))
            return Double(ka.intersection(kb).count) / Double(k)

        case .klDivergence:
            let p = softmax(reference), q = softmax(actual)
            return zip(p, q).reduce(0.0) { sum, pair in
                let (pi, qi) = pair
                return pi > 0 ? sum + pi * log(pi / max(qi, 1e-12)) : sum
            }

        case .l1Mean:
            let total = zip(actual, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) }
            return total / Double(actual.count)
        }
    }

    static func topIndices(_ values: [Double], k: Int) -> [Int] {
        values.indices.sorted { values[$0] > values[$1] }.prefix(k).map { $0 }
    }

    static func softmax(_ values: [Double]) -> [Double] {
        let maxValue = values.max() ?? 0
        let exponentials = values.map { exp($0 - maxValue) }   // shifted for stability
        let total = exponentials.reduce(0, +)
        return total > 0 ? exponentials.map { $0 / total } : exponentials
    }

    static func expand(_ flatIndex: Int, shape: [Int]) -> [Int] {
        var remainder = flatIndex
        var coordinate = [Int](repeating: 0, count: shape.count)
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            guard shape[axis] > 0 else { continue }
            coordinate[axis] = remainder % shape[axis]
            remainder /= shape[axis]
        }
        return coordinate
    }
}
