import Foundation

/// The machine a measurement was taken on.
///
/// Every comparison result and every blessed golden is stamped with this.
/// Tolerances are chip-specific by design — see docs/TEST_STRATEGY.md §4.4 —
/// so a number without a `HostEnvironment` attached is not interpretable.
public struct HostEnvironment: Codable, Sendable, Equatable {

    public enum Chip: String, Codable, Sendable {
        case m1, m1Pro, m1Max, m1Ultra
        case m2, m2Pro, m2Max, m2Ultra
        case m3, m3Pro, m3Max
        case m4, m4Pro, m4Max
        case unknown
    }

    public let chip: Chip
    /// Raw `machdep.cpu.brand_string`, kept verbatim so an unrecognised chip
    /// is still diagnosable rather than silently collapsing to `.unknown`.
    public let chipBrandString: String
    public let osVersion: String

    public init(chip: Chip, chipBrandString: String, osVersion: String) {
        self.chip = chip
        self.chipBrandString = chipBrandString
        self.osVersion = osVersion
    }

    public static func current() -> HostEnvironment {
        let brand = sysctlString("machdep.cpu.brand_string") ?? ""
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return HostEnvironment(
            chip: chip(fromBrand: brand),
            chipBrandString: brand,
            osVersion: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
        )
    }

    /// Parses `"Apple M4 Pro"` into `.m4Pro`. Separated from `current()` so it
    /// can be tested against chips this machine is not.
    public static func chip(fromBrand brand: String) -> Chip {
        let s = brand.lowercased()
        for family in ["m1", "m2", "m3", "m4"] where s.contains(family) {
            for variant in ["ultra", "max", "pro"] where s.contains("\(family) \(variant)") {
                return Chip(rawValue: family + variant.capitalized) ?? .unknown
            }
            return Chip(rawValue: family) ?? .unknown
        }
        return .unknown
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        // sysctl reports a size that includes the trailing NUL, so truncate at
        // the first one rather than decoding it into the String.
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
    }
}
