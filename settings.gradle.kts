rootProject.name = "skainet-embedded-vendors"

// Composite build against the sibling SKaiNET checkout. This resolves every
// `sk.ainet.core:*` dependency to the local SKaiNET projects, which carry the merged
// `TargetOptimizer` / pluggable-optimization mechanism the plugin depends on.
//
// Once SKaiNET publishes a release that includes that mechanism (>= the version after
// 0.33.0), drop this `includeBuild` and the plugin resolves the published artifact instead.
includeBuild("../SKaiNET")

include("synaptics-torq")
