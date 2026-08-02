import Testing
@testable import InferenceProbe

/// First tests in the project. Deliberately small — the point is a green ⌘U
/// and a working Test Navigator before Swift day.
@Suite("HostEnvironment")
struct HostEnvironmentTests {

    @Test("this machine's chip is recognised")
    func currentChipIsRecognised() {
        let env = HostEnvironment.current()
        // The failure message carries the raw brand string, so an unrecognised
        // chip tells you what to add rather than just that something is wrong.
        #expect(env.chip != .unknown,
                "unrecognised brand string: \(env.chipBrandString)")
    }

    @Test("os version is populated")
    func osVersionIsPopulated() {
        #expect(!HostEnvironment.current().osVersion.isEmpty)
    }

    /// Parameterized with `@Test(arguments:)` — the same mechanism the
    /// compute-unit matrix will use on Tuesday.
    @Test("brand strings parse to the right chip", arguments: [
        ("Apple M1",       HostEnvironment.Chip.m1),
        ("Apple M1 Pro",   .m1Pro),
        ("Apple M1 Max",   .m1Max),
        ("Apple M2 Ultra", .m2Ultra),
        ("Apple M4 Pro",   .m4Pro),
        ("Intel Core i9",  .unknown),
    ])
    func brandParsing(brand: String, expected: HostEnvironment.Chip) {
        #expect(HostEnvironment.chip(fromBrand: brand) == expected)
    }
}
