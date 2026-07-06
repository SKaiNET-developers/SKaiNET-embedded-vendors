package sk.ainet.vendors.torq

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Torq-**target** graph pass: prune dangling graph outputs down to the real model output.
 *
 * The DSL trace of `moonshineEncoder` records the per-layer RoPE'd Q/K/V tensors as leaf nodes
 * (no consumers), so `func @main` ends up returning **19** tensors: 18 spurious `[1,H,S,D]`
 * attention intermediates + the real `[.., S, dim]` encoder memory. The llvm-cpu backend
 * tolerates the extra results, but the Torq global-layout solver does NOT — the returned
 * attention intermediates force a K materialization layout that trips `MatMulPattern.cpp:57`
 * (proven: dropping them to the single real output removes the assertion).
 *
 * This pass keeps only the leaves that look like the encoder output — output spec rank ≤ 3 with
 * a trailing model dimension [modelDim] — and iteratively removes every other leaf together with
 * any producer left dangling by the removal, until a fixpoint. Purely structural / HW-agnostic in
 * effect, but only *wanted* for the Torq target, so it is registered by [TorqPlugin.install].
 */
public class TorqPruneOutputsPass(
    private val modelDim: Int,
) : GraphOptimizationPass {
    override val name: String = "torq-prune-outputs"

    /** A leaf is a "real" output if it is rank ≤ 3 and its last dim is the model dim. */
    private fun isKept(n: GraphNode): Boolean {
        val sh = n.outputs.firstOrNull()?.shape ?: return false
        return sh.size in 2..3 && sh.last() == modelDim
    }

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        var changed = false
        // Peel dangling leaves until every leaf is a kept output (or nothing left to remove).
        while (true) {
            val leaves = graph.getOutputNodes() // nodes with no outgoing edges
            val drop = leaves.filterNot { isKept(it) }
            if (drop.isEmpty()) break
            for (n in drop) {
                // Remove the node's incoming edges first, then the node itself. Producers that
                // become leaves as a result are revisited on the next iteration.
                graph.edges.filter { it.destination.id == n.id }.forEach { graph.removeEdge(it) }
                graph.removeNode(n)
                changed = true
            }
        }
        return GraphOptimizationResult(graph, changed = changed)
    }
}
