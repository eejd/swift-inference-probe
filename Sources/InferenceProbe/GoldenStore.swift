import CryptoKit
import Foundation

public struct GoldenID: Hashable, Sendable, CustomStringConvertible {
    public let rawValue: String
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public var description: String { rawValue }
}

/// Provenance for one committed reference, written by
/// `python/generate_goldens.py`.
public struct GoldenManifest: Codable, Sendable {
    public struct Tensor: Codable, Sendable {
        public let file: String
        public let shape: [Int]
        public let dtype: String
        public let sha256: String
    }
    public struct Provenance: Codable, Sendable {
        public let chip: String
        public let macos: String
        public let python: String?
        public let torch: String?
        public let torchvision: String?
        public let coremltools: String?
        public let numpy: String?
    }

    public let id: String
    public let distribution: String?
    public let producedBy: Provenance
    public let blessedAt: String
    public let reason: String
    public let input: Tensor
    public let reference: Tensor
    /// Identity is the **weights**, never the `.mlpackage`: F-011 measured the
    /// package to carry fresh UUIDs and a date stamp on every export, so a
    /// package digest would invalidate every golden while the numerics were
    /// unchanged.
    public let modelWeightSha256: [String: String]?
    public let top1: Int?
    /// Set when a golden was re-blessed, so the audit trail survives.
    public let supersedes: [String]?
}

public struct Golden: Sendable {
    public let id: GoldenID
    public let input: [Double]
    public let reference: [Double]
    public let manifest: GoldenManifest
}

public protocol GoldenStore: Sendable {
    func load(_ id: GoldenID) throws -> Golden
    func compare(_ actual: [Double], to id: GoldenID,
                 using policy: TolerancePolicy) throws -> ComparisonResult
    /// Regenerating a golden requires a stated reason, recorded alongside the
    /// environment that produced it. Blessing is a deliberate, audited act —
    /// not a `--fix` flag.
    func bless(_ actual: [Double], as id: GoldenID, reason: String) throws
}

/// Goldens on disk under `Fixtures/`.
///
/// Tensors are raw little-endian float32 with shape and digest in the manifest,
/// rather than `.npy` as SPEC §4.6 first specified — a flat buffer needs no
/// header parser on this side and no dependency on the other.
public struct FileGoldenStore: GoldenStore {

    public enum Failure: Error, CustomStringConvertible {
        case missing(GoldenID, URL)
        case digestMismatch(String, expected: String, actual: String)
        case sizeMismatch(String, expected: Int, actual: Int)
        case blessWithoutReason(GoldenID)

        public var description: String {
            switch self {
            case .missing(let id, let url):
                return """
                no golden '\(id)' at \(url.path)
                Generate with:  cd python && conda activate inferenceprobe && python generate_goldens.py
                """
            case .digestMismatch(let file, let expected, let actual):
                return """
                \(file) does not match its manifest digest — the fixture has \
                been modified or corrupted.
                  expected \(expected)
                  actual   \(actual)
                """
            case .sizeMismatch(let file, let expected, let actual):
                return "\(file) holds \(actual) floats, manifest says \(expected)"
            case .blessWithoutReason(let id):
                return "refusing to bless '\(id)' without a reason"
            }
        }
    }

    public let goldensDirectory: URL
    public let inputsDirectory: URL

    public init(goldensDirectory: URL = Fixtures.goldensDirectory,
                inputsDirectory: URL = Fixtures.inputsDirectory) {
        self.goldensDirectory = goldensDirectory
        self.inputsDirectory = inputsDirectory
    }

    public var availableIDs: [GoldenID] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: goldensDirectory, includingPropertiesForKeys: nil)) ?? []
        return contents
            .filter { $0.pathExtension == "json" && $0.lastPathComponent != "index.json" }
            .map { GoldenID($0.deletingPathExtension().lastPathComponent) }
            .sorted { $0.rawValue < $1.rawValue }
    }

    public func load(_ id: GoldenID) throws -> Golden {
        let manifestURL = goldensDirectory.appendingPathComponent("\(id.rawValue).json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw Failure.missing(id, manifestURL)
        }
        let manifest = try JSONDecoder().decode(
            GoldenManifest.self, from: Data(contentsOf: manifestURL))

        let input = try readTensor(manifest.input,
                                   from: inputsDirectory.appendingPathComponent(manifest.input.file))
        let reference = try readTensor(manifest.reference,
                                       from: goldensDirectory.appendingPathComponent(manifest.reference.file))
        return Golden(id: id, input: input, reference: reference, manifest: manifest)
    }

    public func compare(_ actual: [Double], to id: GoldenID,
                        using policy: TolerancePolicy) throws -> ComparisonResult {
        let golden = try load(id)
        return TensorComparison.compare(actual, to: golden.reference,
                                        using: policy,
                                        shape: golden.manifest.reference.shape)
    }

    public func bless(_ actual: [Double], as id: GoldenID, reason: String) throws {
        guard !reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Failure.blessWithoutReason(id)
        }
        let existing = try? load(id)
        let tensorName = "\(id.rawValue).blessed.f32.bin"
        let tensorURL = goldensDirectory.appendingPathComponent(tensorName)

        let bytes = writeTensor(actual, to: tensorURL)
        let environment = HostEnvironment.current()

        // The previous reason and date are carried forward rather than
        // overwritten. A golden that has been re-blessed three times should say
        // so — that history is the point of requiring a reason at all.
        var supersedes = existing?.manifest.supersedes ?? []
        if let existing {
            supersedes.append("\(existing.manifest.blessedAt): \(existing.manifest.reason)")
        }

        let manifest = GoldenManifest(
            id: id.rawValue,
            distribution: existing?.manifest.distribution,
            producedBy: .init(chip: environment.chipBrandString,
                              macos: environment.osVersion,
                              python: nil, torch: nil, torchvision: nil,
                              coremltools: nil, numpy: nil),
            blessedAt: ISO8601DateFormatter().string(from: Date()),
            reason: reason,
            input: existing?.manifest.input
                ?? .init(file: "", shape: [], dtype: "float32", sha256: ""),
            reference: .init(file: tensorName,
                             shape: existing?.manifest.reference.shape ?? [actual.count],
                             dtype: "float32", sha256: bytes),
            modelWeightSha256: existing?.manifest.modelWeightSha256,
            top1: actual.indices.max(by: { actual[$0] < actual[$1] }),
            supersedes: supersedes)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest)
            .write(to: goldensDirectory.appendingPathComponent("\(id.rawValue).json"))
    }

    // MARK: - Tensor IO

    func readTensor(_ descriptor: GoldenManifest.Tensor, from url: URL) throws -> [Double] {
        let data = try Data(contentsOf: url)
        let floats = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float32.self))
        }
        let expectedCount = descriptor.shape.reduce(1, *)
        guard floats.count == expectedCount else {
            throw Failure.sizeMismatch(descriptor.file,
                                       expected: expectedCount, actual: floats.count)
        }
        // Verifying the digest is what makes "the fixture is the contract"
        // enforceable rather than aspirational — a silently truncated or
        // half-written tensor would otherwise look like a numerical failure.
        let actualDigest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard actualDigest == descriptor.sha256 else {
            throw Failure.digestMismatch(descriptor.file,
                                         expected: descriptor.sha256, actual: actualDigest)
        }
        return floats.map(Double.init)
    }

    @discardableResult
    func writeTensor(_ values: [Double], to url: URL) -> String {
        let data = values.map(Float32.init).withUnsafeBufferPointer { Data(buffer: $0) }
        try? data.write(to: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
