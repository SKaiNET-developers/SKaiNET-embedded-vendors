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
 * Torq-**target** mixed-precision pass that reproduces the numeric recipe the vendor's own
 * Moonshine encoder uses on this NPU (observed in `encoder.vmfb`:
 * `matmul_..._bf16xbf16xf32` + `reduction_..._f32`):
 *
 *  - **Residual stream + LayerNorm + softmax edges stay f32** — the Torq NPU computes a bf16
 *    LayerNorm variance as NaN, so norms/reductions must run in f32.
 *  - **Every matmul takes bf16 inputs** (weights + activations cast to bf16 at the matmul).
 *  - **2D projection / FFN matmuls accumulate to f32** (their result feeds the f32 LayerNorm).
 *  - **The whole attention interior is bf16** — the QK/AV batched matmuls AND the scale/softmax
 *    between them. The NPU "CSS program" for attention cannot allocate f32 batched matmul
 *    results or f32 batched softmax reductions, and bf16 attention is numerically fine.
 *
 * Concretely: any float tensor with an attention head-group leading dim (rank≥3 or the softmax
 * `[g,seq]` reductions, `dim0 ∈ {2,4,8}`) is the attention interior → bf16; everything else stays
 * f32; matmul inputs are bf16; and a `convert` is spliced onto every edge that crosses a dtype
 * boundary. HW-specific → Torq plugin, applied on the FP32 trace (ENC_DTYPE=FP32 ENC_MATMUL_BF16=1).
 */
class TorqMatmulBf16Pass : GraphOptimizationPass {
    override val name: String = "torq-matmul-bf16"

    private val headGroups = setOf(2, 4, 8)

    private fun isF32(dt: String) = dt.uppercase().let { it == "F32" || it == "FP32" || it == "FLOAT32" }

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val matmulIds = graph.nodes.filter { it.operationName.lowercase() == "matmul" }.map { it.id }.toSet()

        // Attention interior = float tensors carrying a head-group batch dim.
        fun isAttn(n: GraphNode): Boolean {
            val o = n.outputs.firstOrNull() ?: return false
            if (!isF32(o.dtype)) return false
            val sh = o.shape ?: return false
            val d0 = sh.getOrNull(0) ?: return false
            return (sh.size >= 3 && d0 in headGroups) || (sh.size == 2 && d0 in headGroups)
        }
        val attnIds = graph.nodes.filter { isAttn(it) }.map { it.id }.toSet()
        if (matmulIds.isEmpty() && attnIds.isEmpty()) return GraphOptimizationResult(graph, changed = false)

        // Desired dtype of each node's output j.
        fun outDt(n: GraphNode, j: Int): String {
            val d = n.outputs[j].dtype
            return if (n.id in attnIds && isF32(d)) "BF16" else d
        }
        // Desired dtype of node n's input i.
        fun inDt(n: GraphNode, spec: TensorSpec): String {
            if (!isF32(spec.dtype)) return spec.dtype
            return if (n.id in matmulIds || n.id in attnIds) "BF16" else spec.dtype
        }

        val newGraph = DefaultComputeGraph()
        // Rebuild every node with its retyped input/output specs.
        for (n in graph.nodes) {
            val newIns = n.inputs.map { it.copy(dtype = inDt(n, it)) }
            val newOuts = n.outputs.mapIndexed { j, o -> if (n.id in attnIds && isF32(o.dtype)) o.copy(dtype = "BF16") else o }
            newGraph.addNode(GraphNode(n.id, n.operation, newIns, newOuts))
        }
        fun nodeById(id: String): GraphNode? = newGraph.nodes.firstOrNull { it.id == id }

        // Reconnect every edge, splicing a `convert` wherever producer-out dtype != consumer-in dtype.
        var conv = 0
        for (e in graph.edges) {
            val src = nodeById(e.source.id) ?: continue
            val dst = nodeById(e.destination.id) ?: continue
            val pOut = outDt(e.source, e.sourceOutputIndex)
            val cInSpec = dst.inputs[e.destinationInputIndex]
            val cIn = cInSpec.dtype
            if (pOut == cIn) {
                newGraph.addEdge(e.copy(source = src, destination = dst, tensorSpec = cInSpec))
                continue
            }
            // producer emits pOut, consumer wants cIn -> insert convert.
            val srcSpec = src.outputs[e.sourceOutputIndex]
            val cvtIn = srcSpec // dtype pOut
            val cvtOut = cInSpec // dtype cIn
            val cvtNode = GraphNode("mp_convert_${conv++}", GenericOperation("convert", emptyMap()), listOf(cvtIn), listOf(cvtOut))
            newGraph.addNode(cvtNode)
            newGraph.addEdge(GraphEdge("e_${e.source.id}_${e.sourceOutputIndex}__${cvtNode.id}_0", src, cvtNode, e.sourceOutputIndex, 0, cvtIn))
            newGraph.addEdge(GraphEdge("e_${cvtNode.id}_0__${dst.id}_${e.destinationInputIndex}", cvtNode, dst, 0, e.destinationInputIndex, cvtOut))
        }
        return GraphOptimizationResult(newGraph, changed = true)
    }
}
