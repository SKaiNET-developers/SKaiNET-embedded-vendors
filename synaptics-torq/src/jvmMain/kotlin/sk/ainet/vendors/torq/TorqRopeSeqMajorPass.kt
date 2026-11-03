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
 * Torq-**target** pass: rewrite the self-attention **K** path so RoPE is applied SEQ-major.
 *
 * Why: the Torq global-layout solver trips `MatMulPattern.cpp:57` when the QK^T's K operand
 * (`[H,D,Sk]`) is produced by transposing a *head-major* tensor. The moonshine attention reshapes
 * the K projection to seq-major `[Sk,H,D]`, then `permute[1,0,2]`→head-major to run RoPE; the
 * tiling pass then transposes back to build K^T — and that transpose-sourced seq-major K poisons
 * the solver. The hand-written enc6 (no RoPE) keeps K reshape-sourced seq-major and compiles.
 * Proven by isolation (scratchpad/min_oproj*.mlir): seq-major RoPE'd K^T compiles; head-major
 * K^T does not.
 *
 * This pass **sinks the head-transpose below RoPE**: it finds the `permute[1,0,2]` on the K path
 * (seq-major `[Sk,H,D]` → head-major `[H,Sk,D]`, feeding the RoPE that reaches the SDPA's K input),
 * rebuilds every op between that permute and the SDPA in seq-major (swap the leading two dims;
 * reshape any `[Sk,D]` cos/sin broadcast operand to `[Sk,1,D]`), and re-inserts a single
 * `permute[1,0,2]` at the SDPA boundary. Net effect: RoPE runs seq-major, and the downstream
 * TorqAttentionTilingPass builds a reshape-sourced seq-major K^T. HW-agnostic in output (still
 * portable StableHLO); registered for the "torq" target only. Runs BEFORE TorqAttentionTilingPass.
 *
 * Only the K path is rewritten: Q stays head-major (a head-major Q with a seq-sourced K^T
 * compiles — v4b/v4f), and V has no RoPE and needs no seq-major layout.
 */
public class TorqRopeSeqMajorPass : GraphOptimizationPass {
    override val name: String = "torq-rope-seq-major"

    // Ops on the K RoPE path whose semantics are unchanged by swapping the leading two dims
    // (they act on trailing dims / broadcast), so the head-transpose can be sunk past them. All
    // rank-3 [H,Sk,D]; the batch `unsqueeze` right below the SDPA is a rank-changing BOUNDARY that
    // is NOT rebuilt (it stays after the restore permute).
    private val sinkableAlways = setOf(
        "convert", "multiply", "add", "subtract", "reshape", "slice", "narrow", "concat", "concatenate",
    )

    /** An op the head-transpose can be sunk past (commutes with a leading-two-dim swap). Internal
     * `unsqueeze`/`squeeze` (dim ≥ 2, e.g. the rotate-half pair-axis) qualify; the batch
     * `unsqueeze`/`squeeze` (dim 0/1) does not — it is the SDPA boundary, handled separately. */
    private fun isSinkable(n: GraphNode): Boolean {
        val op = n.operationName.lowercase()
        if (op in sinkableAlways) return true
        if (op == "unsqueeze" || op == "squeeze") {
            val dim = (n.operation.parameters["dim"] as? Number)?.toInt() ?: return false
            return dim >= 2
        }
        return false
    }

    private fun isHeadTranspose(n: GraphNode): Boolean {
        if (n.operationName.lowercase() != "permute" && n.operationName.lowercase() != "transpose") return false
        val axes = (n.operation.parameters["axes"] as? List<*>)?.map { (it as Number).toInt() }
        val inSh = n.inputs.firstOrNull()?.shape
        // [1,0,2] on a rank-3 [Sk,H,D] -> [H,Sk,D].
        return axes == listOf(1, 0, 2) && inSh != null && inSh.size == 3
    }

    /** swap the leading two dims of a shape (head<->seq). */
    private fun swap01(sh: List<Int>?): List<Int>? =
        if (sh == null || sh.size < 2) sh else listOf(sh[1], sh[0]) + sh.drop(2)

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val dbg = System.getenv("ROPE_SEQ_DEBUG") == "1"
        val sdpas = graph.nodes.filter { it.operationName.lowercase() == "scaleddotproductattention" }
        if (dbg) println("[rope-seq] SDPA nodes=${sdpas.size} (all ops: ${graph.nodes.map { it.operationName }.distinct().take(40)})")
        if (sdpas.isEmpty()) return GraphOptimizationResult(graph, changed = false)

        val nodeById = graph.nodes.associateBy { it.id }.toMutableMap()
        // producer of (nodeId, inputIdx) -> (node, outIdx)
        val producerOf = HashMap<Pair<String, Int>, Pair<GraphNode, Int>>()
        for (e in graph.edges) producerOf[e.destination.id to e.destinationInputIndex] = e.source to e.sourceOutputIndex

        val newGraph = DefaultComputeGraph()
        graph.nodes.forEach { newGraph.addNode(it) }
        graph.edges.forEach { newGraph.addEdge(it) }
        var changed = false
        var uid = 0

        for (sdpa in sdpas) {
            // 1) The K path (SDPA input 1). The batch `unsqueeze`/`squeeze` right below the SDPA is a
            // rank-changing boundary: skip it and rewire ITS rank-3 input; else rewire SDPA.K directly.
            val kProd0 = producerOf[sdpa.id to 1] ?: continue
            val boundaryIsUnsqueeze = kProd0.first.operationName.lowercase() in setOf("unsqueeze", "squeeze")
            val kProd = if (boundaryIsUnsqueeze) (producerOf[kProd0.first.id to 0] ?: continue) else kProd0
            var headPermute: GraphNode? = null
            val visited = HashSet<String>()
            // BFS up over the (possibly branching, e.g. RoPE add) sinkable region until every branch
            // reaches the SAME head-transpose permute.
            val frontier = ArrayDeque<GraphNode>()
            frontier.add(kProd.first)
            val region = LinkedHashSet<GraphNode>()
            var ok = true
            // The region top (kProd) MUST be sinkable; otherwise there's nothing to rewrite.
            if (!isSinkable(kProd.first)) ok = false
            while (ok && frontier.isNotEmpty()) {
                val n = frontier.removeFirst()
                if (!visited.add(n.id)) continue
                if (isHeadTranspose(n)) {
                    if (headPermute == null) headPermute = n
                    else if (headPermute!!.id != n.id) { ok = false; break } // two different permutes -> bail
                    continue
                }
                if (!isSinkable(n)) continue // external operand (cos/sin/mean/weight): leaf, reuse
                region.add(n)
                // Follow ONLY sinkable producers or the head-transpose; external operands stay put.
                for (i in n.inputs.indices) {
                    val p = producerOf[n.id to i] ?: continue
                    if (isHeadTranspose(p.first) || isSinkable(p.first)) frontier.add(p.first)
                }
            }
            val hp = headPermute
            if (dbg) println("[rope-seq] sdpa=${sdpa.id} boundaryUnsq=$boundaryIsUnsqueeze ok=$ok headPermute=${hp?.id} regionSize=${region.size} regionOps=${region.map { it.operationName }}")
            if (!ok || hp == null || region.isEmpty()) continue

            // 2) The seq-major source feeding the head-transpose.
            val seqSrc = producerOf[hp.id to 0] ?: continue // (node, outIdx) producing [Sk,H,D]

            // 3) Rebuild the region in seq-major. Map old node id -> (new node, outIdx) producing seq-major.
            val remap = HashMap<String, Pair<GraphNode, Int>>()
            // The head-transpose's consumers in the region should read the seq-major source directly.
            remap[hp.id] = seqSrc
            // Rebuild in a stable order: iterate until all region nodes are rebuilt (their inputs available).
            val pending = ArrayDeque(region)
            var guard = region.size * region.size + 8
            val newEdges = ArrayList<GraphEdge>()
            val newNodes = ArrayList<GraphNode>()
            while (pending.isNotEmpty() && guard-- > 0) {
                val n = pending.removeFirst()
                // resolve each input's seq-major producer
                val inProds = ArrayList<Pair<GraphNode, Int>>()
                var ready = true
                for (i in n.inputs.indices) {
                    val p = producerOf[n.id to i]
                    if (p == null) { inProds.add(seqSrc); continue } // constant/no-edge input handled below
                    when {
                        p.first.id == hp.id -> inProds.add(seqSrc)
                        remap.containsKey(p.first.id) -> inProds.add(remap[p.first.id]!!)
                        p.first.id in region.map { it.id } -> { ready = false } // producer in region not rebuilt yet
                        else -> inProds.add(p.first to p.second) // outside region (e.g. cos/sin constant): reuse
                    }
                    if (!ready) break
                }
                if (!ready) { pending.addLast(n); continue }

                // Build seq-major specs: swap leading two dims of inputs/outputs.
                val newIns = ArrayList<TensorSpec>()
                for (i in n.inputs.indices) {
                    val orig = n.inputs[i]
                    val prod = inProds[i]
                    val prodShape = prod.first.outputs.getOrNull(prod.second)?.shape
                    // If the producer already yields a seq-major shape (rebuilt), use it; else this is an
                    // outside-region operand (cos/sin/const) -> keep its shape, but insert a size-1 head
                    // dim so a [Sk,D] operand broadcasts against seq-major [Sk,H,D].
                    val insertHead = prod.first.id !in remap.keys && orig.shape != null && orig.shape!!.size == 2
                    val spec = if (insertHead) {
                        val s = prodShape ?: orig.shape!!
                        orig.copy(name = "${orig.name}_sq${uid}", shape = listOf(s[0], 1, s[1]))
                    } else {
                        orig.copy(name = "${orig.name}_sq${uid}", shape = swap01(orig.shape))
                    }
                    uid++
                    newIns.add(spec)
                }
                val newOuts = n.outputs.map { it.copy(name = "${it.name}_sq${uid++}", shape = swap01(it.shape)) }
                // adjust shape-bearing params (reshape newShape, slice indices, concat dim).
                val np = adjustParams(n.operation.parameters, n)
                val nn = GraphNode("${n.id}_seq", GenericOperation(n.operationName, np), newIns, newOuts)
                newNodes.add(nn)
                remap[n.id] = nn to 0
                // for insertHead operands, splice a reshape node
                for (i in n.inputs.indices) {
                    val prod = inProds[i]
                    val needReshape = prod.first.id !in remap.keys && n.inputs[i].shape?.size == 2
                    if (needReshape) {
                        val srcSpec = prod.first.outputs[prod.second]
                        val rsOut = newIns[i]
                        val rs = GraphNode(
                            "${n.id}_bcastfix_${i}_$uid",
                            GenericOperation("reshape", mapOf<String, Any>("newShape" to (rsOut.shape ?: emptyList<Int>()))),
                            listOf(srcSpec), listOf(rsOut),
                        )
                        uid++
                        newNodes.add(rs)
                        newEdges.add(edge(prod.first, prod.second, rs, 0, srcSpec))
                        newEdges.add(edge(rs, 0, nn, i, rsOut))
                    } else {
                        newEdges.add(edge(prod.first, prod.second, nn, i, newIns[i]))
                    }
                }
            }
            if (dbg) println("[rope-seq] sdpa=${sdpa.id} rebuilt=${remap.size - 1}/${region.size} pendingLeft=${pending.size}")
            if (pending.isNotEmpty()) continue // couldn't rebuild cleanly -> skip this SDPA, leave as-is

            // 4) The region's top node (the one whose OLD output fed the SDPA K, i.e. kProd) now has a
            // seq-major rebuild. Append a permute[1,0,2] to restore head-major and rewire SDPA.K to it.
            val topSeq = remap[kProd.first.id] ?: continue
            val seqSpec = topSeq.first.outputs[topSeq.second]
            val headSpec = seqSpec.copy(name = "${seqSpec.name}_hd$uid", shape = swap01(seqSpec.shape))
            uid++
            val restore = GraphNode(
                "${sdpa.id}_kseq_permute",
                GenericOperation("permute", mapOf("axes" to listOf(1, 0, 2))),
                listOf(seqSpec), listOf(headSpec),
            )
            newNodes.add(restore)
            newEdges.add(edge(topSeq.first, topSeq.second, restore, 0, seqSpec))

            // Commit: add new nodes/edges, then rewire the boundary consumer to the restore permute.
            // If an unsqueeze sits below the SDPA, rewire its rank-3 input; else rewire SDPA.K directly.
            newNodes.forEach { newGraph.addNode(it) }
            newEdges.forEach { newGraph.addEdge(it) }
            val (consumerId, consumerIn, consumerSpec) =
                if (boundaryIsUnsqueeze) Triple(kProd0.first.id, 0, kProd0.first.inputs[0])
                else Triple(sdpa.id, 1, sdpa.inputs[1])
            val consumerNode = newGraph.nodes.firstOrNull { it.id == consumerId } ?: continue
            newGraph.edges.filter { it.destination.id == consumerId && it.destinationInputIndex == consumerIn }
                .forEach { newGraph.removeEdge(it) }
            newGraph.addEdge(edge(restore, 0, consumerNode, consumerIn, consumerSpec))
            changed = true
        }

        return GraphOptimizationResult(newGraph, changed = changed)
    }

    private fun edge(src: GraphNode, srcOut: Int, dst: GraphNode, dstIn: Int, spec: TensorSpec) =
        GraphEdge("e_${src.id}_${srcOut}__${dst.id}_${dstIn}_rsm", src, dst, srcOut, dstIn, spec)

    /** Adjust shape-bearing params of an op when its leading two dims are swapped. */
    private fun adjustParams(params: Map<String, Any>, n: GraphNode): Map<String, Any> {
        val out = params.toMutableMap()
        // reshape newShape -> swap leading two
        (params["newShape"] as? List<*>)?.let { ns ->
            val dims = ns.mapNotNull { (it as? Number)?.toInt() }
            if (dims.size >= 2) out["newShape"] = listOf(dims[1], dims[0]) + dims.drop(2)
        }
        // slice start/limit/strides -> swap leading two entries
        for (key in listOf("start_indices", "limit_indices", "strides", "starts", "limits", "start", "limit")) {
            (params[key] as? List<*>)?.let { idx ->
                val v = idx.mapNotNull { (it as? Number)?.toInt() }
                if (v.size >= 2) out[key] = listOf(v[1], v[0]) + v.drop(2)
            }
        }
        // concat/concatenate dim: 0<->1 swap; others unchanged
        for (key in listOf("dim", "dimension", "axis")) {
            (params[key] as? Number)?.toInt()?.let { d ->
                if (d == 0) out[key] = 1 else if (d == 1) out[key] = 0
            }
        }
        // unsqueeze/squeeze dim in the leading region: 0<->1
        return out
    }
}
