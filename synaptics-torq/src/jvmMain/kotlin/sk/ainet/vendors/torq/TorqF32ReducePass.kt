package sk.ainet.vendors.torq

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Torq-**target** numerical workaround: accumulate the LayerNorm reductions in FP32.
 *
 * The Torq NPU computes bf16 `mean` / `variance` / `sum` reductions with bf16
 * accumulation. For the encoder LayerNorm that makes the variance (`mean(centered²)`,
 * a sum of 288 bf16 squares) lose enough precision that `sqrt(var + eps)` goes NaN/Inf
 * for a large fraction of rows, and the whole encoder output collapses. The exact same
 * bf16 StableHLO runs correctly on `llvm-cpu` (which accumulates reductions in f32), so
 * this is a Torq bf16-reduction precision limitation, not a model bug.
 *
 * This pass wraps every float reduction node with `bf16 -> f32` (input) / `f32 -> bf16`
 * (output) converts, so the reduction ITSELF accumulates in f32 while everything around
 * it (matmuls, elementwise) stays bf16. Compile with `--torq-fallback-f32-to-host` so the
 * f32 reductions run on the host and the bf16 compute stays on the NPU.
 *
 * HW-specific → lives in the Torq plugin set (registered via `TargetOptimizers`), NOT in
 * the model or core. The model stays HW-agnostic bf16; llvm-cpu never needs this pass.
 */
class TorqF32ReducePass : GraphOptimizationPass {
    override val name: String = "torq-f32-reduce"

    private val reduceOps = setOf("mean", "variance", "sum")

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        fun isBf16(dt: String) = dt.uppercase() in setOf("BF16", "BFLOAT16")
        val targets = graph.nodes.filter {
            it.operationName.lowercase() in reduceOps &&
                it.outputs.isNotEmpty() && isBf16(it.outputs[0].dtype)
        }
        if (targets.isEmpty()) return GraphOptimizationResult(graph, changed = false)
        val targetIds = targets.map { it.id }.toSet()

        val producerOf = HashMap<Pair<String, Int>, Pair<GraphNode, Int>>()
        for (e in graph.edges) producerOf[e.destination.id to e.destinationInputIndex] = e.source to e.sourceOutputIndex
        val consumersOf = graph.edges.filter { it.source.id in targetIds }.groupBy { it.source.id }

        val newGraph = DefaultComputeGraph()
        for (n in graph.nodes.filter { it.id !in targetIds }) newGraph.addNode(n)

        fun nodeById(id: String): GraphNode? = newGraph.nodes.firstOrNull { it.id == id }

        for (r in targets) {
            val inSpec = r.inputs[0]
            val outSpec = r.outputs[0]
            val f32In = inSpec.copy(name = "${r.id}_f32in", dtype = "FP32")
            val f32Out = outSpec.copy(name = "${r.id}_f32out", dtype = "FP32")
            // bf16 input -> f32
            val convIn = GraphNode("${r.id}_convin", GenericOperation("convert", emptyMap()), listOf(inSpec), listOf(f32In))
            // the reduction itself, now accumulating in f32
            val f32Reduce = GraphNode(r.id, r.operation, listOf(f32In), listOf(f32Out))
            // f32 result -> bf16
            val convOut = GraphNode("${r.id}_convout", GenericOperation("convert", emptyMap()), listOf(f32Out), listOf(outSpec))
            newGraph.addNode(convIn); newGraph.addNode(f32Reduce); newGraph.addNode(convOut)

            // producer(bf16) -> convIn
            producerOf[r.id to 0]?.let { (src, outIdx) ->
                nodeById(src.id)?.let { s ->
                    newGraph.addEdge(GraphEdge("e_${s.id}_${outIdx}__${convIn.id}_0", s, convIn, outIdx, 0, inSpec))
                }
            }
            // convIn -> f32Reduce -> convOut
            newGraph.addEdge(GraphEdge("e_${convIn.id}_0__${f32Reduce.id}_0", convIn, f32Reduce, 0, 0, f32In))
            newGraph.addEdge(GraphEdge("e_${f32Reduce.id}_0__${convOut.id}_0", f32Reduce, convOut, 0, 0, f32Out))
            // rewire the reduction's original consumers to convOut
            for (c in consumersOf[r.id].orEmpty()) {
                nodeById(c.destination.id)?.let { d ->
                    newGraph.addEdge(c.copy(id = "${c.id}_f32r", source = convOut, destination = d, sourceOutputIndex = 0))
                }
            }
        }

        // Re-add every edge that doesn't touch a rewritten reduction.
        for (e in graph.edges) {
            if (e.source.id in targetIds || e.destination.id in targetIds) continue
            val s = nodeById(e.source.id) ?: continue
            val d = nodeById(e.destination.id) ?: continue
            newGraph.addEdge(e.copy(source = s, destination = d))
        }
        return GraphOptimizationResult(newGraph, changed = true)
    }
}
