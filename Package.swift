// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "InferenceProbe",
    platforms: [.macOS(.v15)],
    targets: [
        .target(name: "InferenceProbe"),

        // Tests of the harness itself. The model-validation suites named in
        // docs/SPEC.md — ParityTests, ComputeUnitTests, MetamorphicTests,
        // RobustnessTests, PerformanceTests — arrive on Swift day (Tue 4 Aug),
        // once there is an exported .mlpackage to point them at.
        .testTarget(name: "HarnessTests", dependencies: ["InferenceProbe"]),
    ]
)
