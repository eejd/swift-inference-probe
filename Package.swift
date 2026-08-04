// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "InferenceProbe",
    platforms: [.macOS(.v15)],
    targets: [
        .target(name: "InferenceProbe"),

        // Tests of the harness itself: no model needed, so these always run,
        // including on a fresh clone where the gitignored .mlpackage files are
        // absent (F-011).
        .testTarget(name: "HarnessTests", dependencies: ["InferenceProbe"]),

        // Model-validation suites from docs/SPEC.md §2.2. These need an
        // exported .mlpackage and skip themselves with an actionable message
        // when one is missing. MetamorphicTests and PerformanceTests are the
        // remaining M4 targets.
        .testTarget(name: "ComputeUnitTests", dependencies: ["InferenceProbe"]),
        .testTarget(name: "RobustnessTests", dependencies: ["InferenceProbe"]),
    ]
)
