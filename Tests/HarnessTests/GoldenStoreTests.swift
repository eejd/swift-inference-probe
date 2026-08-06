import Foundation
import Testing
@testable import InferenceProbe

/// `GoldenStore` and the integrity of the committed fixtures.
///
/// Deliberately in `HarnessTests` rather than `ParityTests`: none of this needs
/// a model, so it runs everywhere — including on a CI runner where the
/// gitignored `.mlpackage` files are absent. Verifying that the committed
/// tensors still match their manifests is one of the few checks that is both
/// meaningful and **chip-independent**, which is exactly what a cross-machine
/// gate should consist of (TEST_STRATEGY §4.4).
@Suite("GoldenStore")
struct GoldenStoreTests {

    // MARK: - Fixture integrity

    @Suite("committed fixtures",
           .enabled(if: Fixtures.goldensAvailable,
                    "no goldens — run: cd python && python generate_goldens.py"))
    struct CommittedFixtures {
        let store = FileGoldenStore()

        @Test("every golden loads and its tensors match their recorded digests")
        func integrity() throws {
            let ids = store.availableIDs
            #expect(!ids.isEmpty, "no goldens found")

            for id in ids {
                // `load` recomputes SHA-256 over both tensors and throws on a
                // mismatch, so this is the integrity assertion. A truncated or
                // partially-written fixture fails here as corruption rather
                // than surfacing later as a mysterious numerical failure.
                let golden = try store.load(id)
                let input = try #require(golden.input,
                                         "\(id) has no paired input tensor")
                #expect(input.count == 3 * 224 * 224)
                #expect(golden.reference.count == Fixtures.classCount)
            }
        }

        @Test("every golden records why it was blessed and against which weights")
        func provenance() throws {
            for id in store.availableIDs {
                let manifest = try store.load(id).manifest
                #expect(!manifest.reason.isEmpty,
                        "\(id) has no reason — blessing is an audited act")
                #expect(!manifest.blessedAt.isEmpty)
                // F-011: identity is the weights, never the .mlpackage.
                #expect(manifest.modelWeightSha256?["fp32"] != nil,
                        "\(id) records no fp32 weight digest, so staleness against the model cannot be detected")
            }
        }
    }

    // MARK: - Blessing

    /// A scratch store, so these never touch `Fixtures/`.
    func makeTemporaryStore() throws -> (FileGoldenStore, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("goldenstore-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (FileGoldenStore(goldensDirectory: root, inputsDirectory: root), root)
    }

    @Test("blessing without a reason is refused")
    func blessRequiresReason() throws {
        let (store, root) = try makeTemporaryStore()
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(throws: FileGoldenStore.Failure.self) {
            try store.bless([1, 2, 3], as: GoldenID("anything"), reason: "   ")
        }
    }

    @Test("a blessed golden round-trips and records the host environment")
    func blessRoundTrips() throws {
        let (store, root) = try makeTemporaryStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let id = GoldenID("scratch")
        try store.bless([1.0, 5.0, 2.0], as: id, reason: "first baseline")

        let loaded = try store.load(id)
        #expect(loaded.reference == [1.0, 5.0, 2.0])
        #expect(loaded.manifest.reason == "first baseline")
        #expect(loaded.manifest.top1 == 1, "index of the largest value")
        #expect(loaded.manifest.producedBy.chip == HostEnvironment.current().chipBrandString)
    }

    /// The audit trail is the point of requiring a reason. A golden re-blessed
    /// three times should say so, rather than presenting the newest reason as
    /// though it were the only one.
    @Test("re-blessing carries the previous reason forward instead of overwriting")
    func blessKeepsHistory() throws {
        let (store, root) = try makeTemporaryStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let id = GoldenID("scratch")
        try store.bless([1.0], as: id, reason: "first baseline")
        try store.bless([2.0], as: id, reason: "re-blessed after torch bump")
        try store.bless([3.0], as: id, reason: "re-blessed after OS update")

        let manifest = try store.load(id).manifest
        #expect(manifest.reason == "re-blessed after OS update")
        #expect(manifest.supersedes?.count == 2)
        #expect(manifest.supersedes?.contains { $0.contains("first baseline") } == true)
        #expect(manifest.supersedes?.contains { $0.contains("torch bump") } == true)
    }

    @Test("a tampered tensor is caught by its digest, not mistaken for drift")
    func digestCatchesTampering() throws {
        let (store, root) = try makeTemporaryStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let id = GoldenID("scratch")
        try store.bless([1.0, 2.0, 3.0], as: id, reason: "baseline")

        // Corrupt the tensor while leaving the manifest untouched — the exact
        // shape of a partial write or a bad merge.
        let tensorURL = root.appendingPathComponent("\(id.rawValue).blessed.f32.bin")
        try Data(repeating: 0, count: 12).write(to: tensorURL)

        #expect(throws: FileGoldenStore.Failure.self) {
            _ = try store.load(id)
        }
    }
}
