# skainet-embedded-vendors

Vendor-specific SKaiNET plugins that live **outside** the agnostic SKaiNET core and the model
definitions. Each plugin registers a hardware target's optimizations with the SKaiNET
`TargetOptimizers` registry; an application selects a target and installs the matching plugin
at its composition root.

## `synaptics-torq` — Synaptics Torq NPU (SL2610)

The vendor optimization handler for the `"torq"` target.

- **DAG passes** (registered via `TorqPlugin.install()`): attention head/query-sequence tiling
  and FFN hidden-dim tiling, so the batched matmuls fit the NPU's on-chip layout.
- **Compile-time flags** (`TorqPlugin.compileFlags`): the `torq-compile` knobs the build tool
  applies — notably `--torq-disable-slices`, required to avoid a buffer-aliasing blow-up in
  large graphs.
- **Optional bf16-trace workarounds** (present, not registered by default): `TorqF32ReducePass`,
  `TorqMatmulBf16Pass`.

### Use (from the application / build tool)

```kotlin
import sk.ainet.vendors.torq.TorqPlugin

// once, at the composition root — the app has chosen the Torq target:
TorqPlugin.install()                 // registers the "torq" DAG passes
// ... trace model -> ComputeGraph ...
val tiled = dagPipelineFor(TorqPlugin.TARGET).optimize(graph).graph
val mlir  = toStableHlo(tiled, "encoder").content
// then invoke torq-compile with TorqPlugin.compileFlags (+ -o out.vmfb)
```

The SKaiNET core never names `"torq"`; it only sees the target string and asks the registry
which passes to run. This keeps all Synaptics specifics out of core and out of the model.

### Dependency layering

```
app (e.g. SKaiNET-embedded/sl2610-function-calling)
 ├── moonshine-dsl model  → transitive: skainet core          [HW-agnostic]
 └── synaptics-torq plugin (this repo) → skainet-compile-opt   [the TargetOptimizer API only]
       app calls TorqPlugin.install() once
```

### Build

This module depends on the `TargetOptimizer` / pluggable-optimization mechanism that is newer
than published SKaiNET `0.33.0`. Until a release that includes it ships, build against a local
SKaiNET checkout via the composite build wired in `settings.gradle.kts`:

```bash
# expects ../SKaiNET to be a sibling checkout of a branch that has the pluggable mechanism
./gradlew :synaptics-torq:compileKotlinJvm
```

Once SKaiNET publishes that version, drop the `includeBuild("../SKaiNET")` in
`settings.gradle.kts` and bump the `sk.ainet.core:*` versions in
`synaptics-torq/build.gradle.kts`.

### Migration note

The four `Torq*Pass` classes and the `registerDagPasses("torq")` call currently also exist in
`SKaiNET-transformers/llm-inference/moonshine/src/jvmTest` (the demo's MLIR-dump test). Those
are **superseded** by this plugin. Once the app wires `TorqPlugin.install()` and the encoder
MLIR generation moves out of the model's test, remove the copies from the moonshine test.
