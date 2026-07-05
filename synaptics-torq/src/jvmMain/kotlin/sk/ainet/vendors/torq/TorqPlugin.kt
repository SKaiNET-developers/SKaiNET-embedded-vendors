package sk.ainet.vendors.torq

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.TargetOptimizer
import sk.ainet.compile.opt.TargetOptimizers

/**
 * Synaptics **Torq NPU** target plugin for SKaiNET.
 *
 * This is the vendor-specific optimization handler for the `"torq"` target. It lives OUTSIDE
 * the SKaiNET core and the model definitions: an application/build tool that has chosen to
 * target the Torq NPU calls [install] once at its composition root, and the plugin registers
 * the target's DAG passes with the agnostic [TargetOptimizers] registry. The core never names
 * `"torq"` — it only sees the target string and asks the registry which passes to run.
 *
 * Registered passes (applied during DAG optimization, before StableHLO emission):
 *  - [TorqAttentionTilingPass] — head-group + query-sequence tiling of the attention block so
 *    the batched QK/AV matmuls fit the NPU's on-chip layout.
 *  - [TorqFfnTilingPass] — hidden-dimension tiling of the FFN.
 *
 * Also available (NOT registered by default — bf16-trace numeric workarounds, opt in explicitly
 * where the trace runs bf16 rather than the validated f32 path):
 *  - [TorqF32ReducePass] — accumulate LayerNorm reductions in f32.
 *  - [TorqMatmulBf16Pass] — vendor mixed precision (`bf16×bf16→f32` matmuls).
 *
 * The Torq **compile-time** knobs (passed to `torq-compile`, not part of DAG optimization) are
 * exposed as [compileFlags] so the build tool can apply them consistently — notably
 * `--torq-disable-slices`, which avoids a buffer-aliasing blow-up in large graphs.
 */
public object TorqPlugin {
    /** The target id (matches the iree device name used by `dagPipelineFor` / `toStableHlo`). */
    public const val TARGET: String = "torq"

    /**
     * Register the Torq target's DAG passes with [TargetOptimizers]. Idempotent-ish: calling
     * twice registers twice, so call once at the composition root.
     *
     * @param maxHeadsPerTile   attention head-group tile size (Moonshine-tiny: 4)
     * @param maxQuerySeqPerTile attention query-sequence tile size (Moonshine encoder @165: 83)
     * @param ffnHiddenTile     FFN hidden-dimension tile size (Moonshine-tiny: 288)
     */
    public fun install(
        maxHeadsPerTile: Int = 4,
        maxQuerySeqPerTile: Int = 83,
        ffnHiddenTile: Int = 288,
    ) {
        TargetOptimizers.register(object : TargetOptimizer {
            override val target: String = TARGET
            override fun dagPasses(): List<GraphOptimizationPass> = listOf(
                TorqAttentionTilingPass(
                    maxHeadsPerTile = maxHeadsPerTile,
                    maxQuerySeqPerTile = maxQuerySeqPerTile,
                ),
                TorqFfnTilingPass(hiddenTile = ffnHiddenTile),
            )
            // granularity() intentionally left at the default (null / decompose-all): the Torq
            // compiler rejects `stablehlo.composite`, so there is no fused-op emission to select
            // yet. Correctness for Torq is achieved by matching the vendor's op *structure* in
            // the model emitters (f32 LayerNorm, full-head RoPE) rather than a granularity policy.
        })
    }

    /**
     * Recommended `torq-compile` flags for the SL2610. Applied by the build tool when invoking
     * the vendor compiler — these are compile-time knobs, not DAG passes.
     *
     * `--torq-disable-slices` is required: Torq's slice-based NSS execution mis-aliases buffers
     * in large graphs (LayerNorm/attention output collapses to NaN/1e23 without it).
     */
    public val compileFlags: List<String> = listOf(
        "--torq-hw=SL2610",
        "--torq-target-host-triple=native",
        "--torq-css-qemu",
        "--torq-disable-slices",
    )
}
