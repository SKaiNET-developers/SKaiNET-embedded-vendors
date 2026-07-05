# SKaiNET-embedded-vendors

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

> Note: the SL2610 v2.0.0 compiler (with `--torq-disable-slices`) compiles the Moonshine
> encoder **without** the attention/FFN tiling — the tiling once worked around a
> `MatMulPattern.cpp:57` assertion that the newer compiler no longer hits. The passes are kept
> here for other models / older toolchains, but Moonshine's Torq path no longer requires them.

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

### Consume

Published to Maven Central under `sk.ainet.vendors`:

```kotlin
dependencies {
    implementation("sk.ainet.vendors:synaptics-torq:0.1.0")
}
```

### Build

Self-contained — depends on the published `sk.ainet.core:*` artifacts (Maven Central, version
pinned in `gradle/libs.versions.toml`), which include the `TargetOptimizer` /
pluggable-optimization mechanism.

```bash
./gradlew :synaptics-torq:compileKotlinJvm
```

To develop against a local SKaiNET checkout instead, uncomment `includeBuild("../SKaiNET")` in
`settings.gradle.kts` (Gradle substitutes the `sk.ainet.core:*` deps with the local projects).

### Publish

Set up the same way as the SKaiNET core: the [vanniktech maven-publish] plugin driven by
`gradle.properties` (root POM + `mavenCentralPublishing` / `signAllPublications`) and per-module
`POM_ARTIFACT_ID` / `POM_NAME`. Bump `VERSION_NAME` in `gradle.properties` for a release.

```bash
# proof-of-build, no signing key needed:
./gradlew publishToMavenLocal -PsignAllPublications=false

# real release (needs Sonatype creds + GPG signing key in the environment):
./gradlew publishToMavenCentral            # staged; release manually on Sonatype
./gradlew publishAndReleaseToMavenCentral  # staged + auto-released
```

Credentials/signing are supplied at CI time via the standard vanniktech env/properties
(`mavenCentralUsername`, `mavenCentralPassword`, `signingInMemoryKey`, `signingInMemoryKeyPassword`);
they are never committed. `signAllPublications=true` means a local publish without a configured
GPG signatory fails on the `sign*` task by design — pass `-PsignAllPublications=false` to test the
artifact assembly.

[vanniktech maven-publish]: https://vanniktech.github.io/gradle-maven-publish-plugin/

### Status

The four `Torq*Pass` classes and the `registerDagPasses("torq")` call have been **removed** from
`SKaiNET-transformers/llm-inference/moonshine` (commit `575bc97` there) — that test now emits
portable, HW-agnostic StableHLO, and this plugin is the sole home for the Torq passes. An
app/build tool that targets Torq calls `TorqPlugin.install()` and applies the tiling during its
own trace→graph pipeline (the moonshine demo currently doesn't need the tiling — see the note
above).
