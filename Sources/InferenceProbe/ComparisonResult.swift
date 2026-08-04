import Foundation

/// The outcome of comparing two tensors under a `TolerancePolicy`.
///
/// The design requirement from docs/SPEC.md §4.3 is that a failure **names the
/// offending thing** rather than reporting that outputs differ. So this carries
/// the measured value next to its threshold, the coordinate of the worst
/// element, the host chip, and — when the Python localizer has run — the ops
/// responsible (F-012).
public struct ComparisonResult: Sendable {

    public struct CriterionOutcome: Sendable {
        public let criterion: TolerancePolicy.Criterion
        public let measured: Double
        public let threshold: Double
        public let passed: Bool
    }

    public let outcomes: [CriterionOutcome]
    public let environment: HostEnvironment
    /// Coordinate of the largest absolute deviation, expanded against the
    /// tensor shape when one is known.
    public let worstIndex: [Int]?
    public let worstDeviation: Double
    /// Populated from `Fixtures/models/op_localization.json`, produced by
    /// `python/localize.py`. Meaningless without the `atol` that produced it,
    /// so `suspectOpsAtol` travels with it (F-012).
    public let suspectOps: [String]
    public let suspectOpsAtol: Double?

    public var passed: Bool { outcomes.allSatisfy(\.passed) }

    public init(outcomes: [CriterionOutcome],
                environment: HostEnvironment,
                worstIndex: [Int]? = nil,
                worstDeviation: Double = 0,
                suspectOps: [String] = [],
                suspectOpsAtol: Double? = nil) {
        self.outcomes = outcomes
        self.environment = environment
        self.worstIndex = worstIndex
        self.worstDeviation = worstDeviation
        self.suspectOps = suspectOps
        self.suspectOpsAtol = suspectOpsAtol
    }

    public var failureDescription: String {
        guard !passed else { return "passed" }
        var lines: [String] = []
        for outcome in outcomes where !outcome.passed {
            let relation = outcome.criterion.higherIsBetter ? "below floor" : "over limit"
            lines.append("  ✗ \(outcome.criterion.name): "
                         + "measured \(fmt(outcome.measured)) \(relation) "
                         + "\(fmt(outcome.threshold))")
        }
        for outcome in outcomes where outcome.passed {
            lines.append("  ✓ \(outcome.criterion.name): \(fmt(outcome.measured))")
        }
        if let worstIndex {
            lines.append("  worst element at [\(worstIndex.map(String.init).joined(separator: ", "))]"
                         + " — deviation \(fmt(worstDeviation))")
        }
        if !suspectOps.isEmpty {
            let atol = suspectOpsAtol.map { " (at atol \(fmt($0)))" } ?? ""
            lines.append("  suspect ops\(atol): \(suspectOps.joined(separator: ", "))")
        }
        // The chip is not decoration. A golden blessed on one chip is not
        // assumed valid on another (SPEC §4.6), so the failure has to say
        // which machine produced it.
        lines.append("  host: \(environment.chipBrandString) / macOS \(environment.osVersion)")
        return lines.joined(separator: "\n")
    }

    private func fmt(_ value: Double) -> String {
        if value != 0 && (abs(value) < 1e-4 || abs(value) >= 1e6) {
            return String(format: "%.4e", value)
        }
        return String(format: "%.6f", value)
    }
}
