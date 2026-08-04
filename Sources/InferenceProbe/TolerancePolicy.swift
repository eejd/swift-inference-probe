import Foundation

/// The comparison vocabulary.
///
/// There is no single right way to compare two tensors, because different
/// criteria are blind to different failures: cosine similarity sees direction
/// and misses a uniform rescale, top-k sees the user-visible answer and misses
/// everything below it, L1 sees magnitude and is diluted by a large tensor
/// where only one element is badly wrong. F-008 is the concrete case — cosine
/// stayed above 0.9995 across every input while 1.5% of them changed class.
///
/// So a test states which criterion it is defending against, and why.
/// See docs/TEST_STRATEGY.md §4.1.
public struct TolerancePolicy: Sendable {

    public enum Criterion: Sendable, Equatable, Codable {
        /// `|a - b| <= atol`, elementwise.
        case absolute(atol: Double)
        /// `|a - b| <= rtol * |b|`, elementwise.
        case relative(rtol: Double)
        /// `|a - b| <= atol + rtol * |b|`, elementwise. The numpy convention.
        case mixed(atol: Double, rtol: Double)
        case cosineSimilarity(min: Double)
        case topKAgreement(k: Int, minOverlap: Double)
        /// Over softmax of both tensors, in nats.
        case klDivergence(maxNats: Double)
        /// Mean `|a - b|`. Directly comparable to the figures in coremltools
        /// issue #2625 and to F-006 through F-009.
        case l1Mean(max: Double)

        /// Whether a larger measured value is better. Needed to render a
        /// failure as "0.9987 below the 0.9990 floor" rather than backwards.
        public var higherIsBetter: Bool {
            switch self {
            case .cosineSimilarity, .topKAgreement: return true
            case .absolute, .relative, .mixed, .klDivergence, .l1Mean: return false
            }
        }

        public var name: String {
            switch self {
            case .absolute:         return "absolute"
            case .relative:         return "relative"
            case .mixed:            return "mixed"
            case .cosineSimilarity: return "cosineSimilarity"
            case .topKAgreement(let k, _): return "top\(k)Agreement"
            case .klDivergence:     return "klDivergence"
            case .l1Mean:           return "l1Mean"
            }
        }

        /// The number the measurement is checked against.
        public var threshold: Double {
            switch self {
            case .absolute(let atol):                return atol
            case .relative(let rtol):                return rtol
            case .mixed(let atol, _):                return atol
            case .cosineSimilarity(let min):         return min
            case .topKAgreement(_, let minOverlap):  return minOverlap
            case .klDivergence(let maxNats):         return maxNats
            case .l1Mean(let max):                   return max
            }
        }
    }

    /// All criteria must pass.
    public let criteria: [Criterion]

    /// Non-optional by design. A policy cannot be constructed without stating
    /// why these thresholds and not others, which makes "document the
    /// threshold" a property of the type system rather than a review
    /// convention. Do not make this optional — see docs/TEST_STRATEGY.md §4.3
    /// and the standing instruction in CLAUDE.md.
    public let rationale: String

    /// Criteria that replace the base set on a specific chip. Empty means the
    /// base criteria apply everywhere. Tolerances are chip-specific by design
    /// (TEST_STRATEGY §4.4), and F-010 turned that from an inherited assumption
    /// into a measured one by eliminating every other explanation for the
    /// #2625 discrepancy.
    public let chipOverrides: [HostEnvironment.Chip: [Criterion]]

    public init(criteria: [Criterion],
                rationale: String,
                chipOverrides: [HostEnvironment.Chip: [Criterion]] = [:]) {
        // An empty policy type-checks but asserts nothing, which is worse than
        // no test at all because it reports green. Fail loudly at construction.
        precondition(!criteria.isEmpty,
                     "TolerancePolicy with no criteria asserts nothing")
        precondition(!rationale.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                     "TolerancePolicy.rationale must say why these thresholds")
        self.criteria = criteria
        self.rationale = rationale
        self.chipOverrides = chipOverrides
    }

    public func criteria(for environment: HostEnvironment) -> [Criterion] {
        chipOverrides[environment.chip] ?? criteria
    }
}
