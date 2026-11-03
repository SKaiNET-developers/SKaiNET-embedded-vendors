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
 * Torq-**target** graph-lowering pass — rewrites each `scaledDotProductAttention`
 * node into a subgraph of **standard ops** (reshape / slice / transpose / matmul /
 * softmax / concatenate) shaped the way the Torq NPU compiler accepts:
 *   - fold `[1,H,S,D]` → `[H,S,D]` (Torq rejects 4D-batched matmul);
 *   - transpose K so QK^T is a standard `A[M,K]·B[K,N]` matmul; and
 *   - split heads into groups of ≤[maxHeadsPerTile] (>4 heads overflow NPU SRAM).
 *
 * The output is still **portable StableHLO** (it runs on llvm-cpu too) — this pass is
 * target-aware but the shared IR emitter and the model definition stay HW-agnostic.
 * It lives OUTSIDE core and plugs into `TargetOptimizers` for the `"torq"` target, so
 * the compiler core carries no Torq knowledge.
 *
 * On Torq **stable v2.0.0** head-grouping alone is not enough: a single attention layer
 * WITH its projections overflows LRAM at full seq length. So each head-group is ALSO
 * **query-sequence tiled** (flash-attention style): the query positions are split into
 * chunks of ≤[maxQuerySeqPerTile], each chunk attends to the full K/V, and the per-chunk
 * outputs are concatenated back along the sequence axis. This shrinks the `[g,Sq,Sk]`
 * scores intermediate per dispatch so the layer fits (verified: the full 6-layer encoder
 * compiles on v2.0.0 with `maxHeadsPerTile=4`, `maxQuerySeqPerTile=83`).
 */
class TorqAttentionTilingPass(
    private val maxHeadsPerTile: Int = 4,
    private val maxQuerySeqPerTile: Int = 83,
) : GraphOptimizationPass {
    override val name: String = "torq-attention-tiling"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val sdpas = graph.nodes.filter { it.operationName.lowercase() == "scaleddotproductattention" }
        if (sdpas.isEmpty()) return GraphOptimizationResult(graph, changed = false)

        val newGraph = DefaultComputeGraph()
        // Carry over every non-SDPA node unchanged.
        val sdpaIds = sdpas.map { it.id }.toSet()
        val kept = graph.nodes.filter { it.id !in sdpaIds }
        for (n in kept) newGraph.addNode(n)

        // Producer of each (nodeId, inputIndex): (source node, source output index).
        val producerOf = HashMap<Pair<String, Int>, Pair<GraphNode, Int>>()
        for (e in graph.edges) producerOf[e.destination.id to e.destinationInputIndex] = e.source to e.sourceOutputIndex
        // Consumers of each SDPA output.
        val consumersOf = graph.edges.filter { it.source.id in sdpaIds }.groupBy { it.source.id }

        for (sdpa in sdpas) {
            val q = sdpa.inputs[0]; val k = sdpa.inputs[1]; val v = sdpa.inputs[2]
            val out = sdpa.outputs[0]
            val elem = out.dtype
            // Expect [1, H, S, D].
            val h = q.shape!![1]; val sq = q.shape!![2]; val d = q.shape!![3]; val sk = k.shape!![2]
            val scale = (sdpa.operation.parameters["scale"] as? Number)?.toFloat()
                ?: (1.0f / kotlin.math.sqrt(d.toFloat()))
            val base = sdpa.id
            var counter = 0
            fun spec(vararg dims: Int) = TensorSpec("${base}_t${counter++}", dims.toList(), elem)
            val nodes = mutableListOf<GraphNode>()
            val edges = mutableListOf<Triple<Pair<GraphNode, Int>, GraphNode, Int>>() // (src,outIdx)->(dst,inIdx)
            fun op(opName: String, params: Map<String, Any>, ins: List<TensorSpec>, o: TensorSpec): GraphNode {
                val n = GraphNode("${base}_${opName}_${counter++}", GenericOperation(opName, params), ins, listOf(o))
                nodes += n; return n
            }

            // Fold [1,H,S,D] -> [H,S,D], but AVOID a redundant round-trip.
            // The model produces each SDPA input by reshaping an existing [H,S,D] up to
            // [1,H,S,D]; naively reshaping it back down yields [H,S,D]->[1,H,S,D]->[H,S,D],
            // which breaks Torq's matmul layout inference (the QK output then makes the AV
            // matmul trip MatMulPattern.cpp:57). So when the input's producer is exactly a
            // reshape from [H,seq,D], consume that [H,seq,D] source directly; otherwise emit
            // the fold reshape as before. Returns (source (node,outIdx), source [H,seq,D] spec).
            fun resolveFold(inputIdx: Int, seqLen: Int): Pair<Pair<GraphNode, Int>, TensorSpec> {
                val prod = producerOf[sdpa.id to inputIdx]
                if (prod != null) {
                    val pn = prod.first
                    val ps = pn.inputs.getOrNull(0)?.shape
                    // The model re-adds the batch-of-1 via `unsqueeze` (or `reshape`) over an
                    // existing [H,seq,D]; bypass it and consume that source directly.
                    if (pn.operationName.lowercase() in setOf("unsqueeze", "reshape") && pn.inputs.size == 1 &&
                        ps != null && ps.size == 3 && ps[0] == h && ps[1] == seqLen && ps[2] == d
                    ) {
                        producerOf[pn.id to 0]?.let { gp -> return gp to pn.inputs[0] }
                    }
                }
                // Fallback: emit the [1,H,seq,D] -> [H,seq,D] fold reshape.
                val fs = spec(h, seqLen, d)
                val rn = op("reshape", emptyMap(), listOf(sdpa.inputs[inputIdx]), fs)
                prod?.let { edges += Triple(it, rn, 0) }
                return (rn to 0) to fs
            }
            val (qSrc, q3s) = resolveFold(0, sq)
            val (kSrc, k3s) = resolveFold(1, sk)
            val (vSrc, v3s) = resolveFold(2, sk)

            // Transpose the FULL K [H,Sk,D] -> [H,D,Sk] ONCE, before slicing head-groups —
            // ROUTED THROUGH the [Sk,H,D] ordering:
            //   k3s [H,Sk,D] --[1,0,2]--> [Sk,H,D] --[1,2,0]--> [H,D,Sk]
            // Two layout facts, both proven by controlled experiments on Torq v2.0.0:
            //  (root cause #1) the transpose must precede the head-group slice, not follow it
            //    (transpose-then-slice OK, slice-then-transpose FAIL); and
            //  (root cause #2) the final [H,D,Sk] transpose must be a [1,2,0] FROM a [Sk,H,D]
            //    tensor — a direct [0,2,1] on the [H,Sk,D] source mis-seeds Torq's GLOBAL layout
            //    solver and trips MatMulPattern.cpp:57 on the downstream AV matmul, even though
            //    the AV operands' local producer chains are byte-identical to a compiling graph.
            // [1,2,0]-from-[Sk,H,D] is exactly how the known-good hand-written enc6 builds K.
            // The extra [1,0,2] hop makes this robust to RoPE (which leaves K in [H,Sk,D]) and
            // to the no-RoPE case alike; without RoPE the paired [1,0,2] transposes canonicalize
            // away to the single [1,2,0].
            // If TorqRopeSeqMajorPass ran, the SDPA's K producer is `permute[1,0,2](seqK)` where seqK
            // is the RoPE'd K already in seq-major [Sk,H,D] (reshape-sourced). Consume seqK directly and
            // build K^T with a SINGLE [1,2,0] — the reshape-sourced seq-major K^T the Torq layout solver
            // accepts (a head-major-sourced K^T trips MatMulPattern.cpp:57). Otherwise fall back to the
            // route-through-[Sk,H,D] double transpose.
            fun permAxes(n: GraphNode): List<Int>? =
                (n.operation.parameters["permutation"] ?: n.operation.parameters["axes"])
                    .let { it as? List<*> }?.map { (it as Number).toInt() }
            val kSeq: Pair<Pair<GraphNode, Int>, TensorSpec>? = run {
                val p = kSrc.first
                val op = p.operationName.lowercase()
                if ((op == "permute" || op == "transpose") && permAxes(p) == listOf(1, 0, 2) &&
                    p.inputs.firstOrNull()?.shape?.size == 3
                ) producerOf[p.id to 0]?.let { it to p.inputs[0] } else null
            }
            val kTS = spec(h, d, sk)
            val kT = if (kSeq != null) {
                val (seqSrc, seqSpec) = kSeq // seqSpec = [Sk,H,D] seq-major RoPE'd K
                op("transpose", mapOf("permutation" to listOf(1, 2, 0)), listOf(seqSpec), kTS)
                    .also { edges += Triple(seqSrc, it, 0) }
            } else {
                val kShdS = spec(sk, h, d)
                val kShd = op("transpose", mapOf("permutation" to listOf(1, 0, 2)), listOf(k3s), kShdS)
                    .also { edges += Triple(kSrc, it, 0) }
                op("transpose", mapOf("permutation" to listOf(1, 2, 0)), listOf(kShdS), kTS)
                    .also { edges += Triple(kShd to 0, it, 0) }
            }

            val groupOuts = mutableListOf<Pair<GraphNode, Int>>() // (node, headCount)
            var s = 0
            while (s < h) {
                val e = minOf(s + maxHeadsPerTile, h); val g = e - s
                fun sliceP(start: Int, limit: Int, s1: Int, s2: Int) = mapOf(
                    "start_indices" to listOf(start, 0, 0),
                    "limit_indices" to listOf(limit, s1, s2),
                    "strides" to listOf(1, 1, 1),
                )
                val qgS = spec(g, sq, d); val vgS = spec(g, sk, d)
                val qg = op("slice", sliceP(s, e, sq, d), listOf(q3s), qgS).also { edges += Triple(qSrc, it, 0) }
                val vg = op("slice", sliceP(s, e, sk, d), listOf(v3s), vgS).also { edges += Triple(vSrc, it, 0) }
                // Slice the pre-transposed K [H,D,Sk] into this head-group -> [g,D,Sk].
                val ktS = spec(g, d, sk)
                val kt = op(
                    "slice",
                    mapOf(
                        "start_indices" to listOf(s, 0, 0),
                        "limit_indices" to listOf(e, d, sk),
                        "strides" to listOf(1, 1, 1),
                    ),
                    listOf(kTS), ktS,
                ).also { edges += Triple(kT to 0, it, 0) }

                // Query-sequence tiling (flash-style): split the Sq query positions into
                // even chunks of ≤maxQuerySeqPerTile; each chunk attends to the full K/V.
                val nSeq = (sq + maxQuerySeqPerTile - 1) / maxQuerySeqPerTile
                val seqTile = (sq + nSeq - 1) / nSeq
                val seqOuts = mutableListOf<GraphNode>()
                var qs = 0
                while (qs < sq) {
                    val qe = minOf(qs + seqTile, sq); val ql = qe - qs
                    // slice Qg [g,Sq,D] -> [g,ql,D] along the sequence axis
                    val qcS = spec(g, ql, d)
                    val qc = op(
                        "slice",
                        mapOf(
                            "start_indices" to listOf(0, qs, 0),
                            "limit_indices" to listOf(g, qe, d),
                            "strides" to listOf(1, 1, 1),
                        ),
                        listOf(qgS), qcS,
                    ).also { edges += Triple(qg to 0, it, 0) }
                    // scores = Qc @ Kt  (torq-friendly batching[0] contracting[2]x[1])
                    val scS = spec(g, ql, sk)
                    val sc = op("matmul", emptyMap(), listOf(qcS, ktS), scS)
                        .also { edges += Triple(qc to 0, it, 0); edges += Triple(kt to 0, it, 1) }
                    // scale: multiply by a full-shape splat of 1/sqrt(head_dim)
                    // (a rank-0 scalar mis-prints as tensor<xbf16>).
                    val scaleCS = spec(g, ql, sk)
                    val scaleC = op("splat_constant", mapOf("value" to scale), emptyList(), scaleCS)
                    val scdS = spec(g, ql, sk)
                    val scd = op("multiply", emptyMap(), listOf(scS, scaleCS), scdS)
                        .also { edges += Triple(sc to 0, it, 0); edges += Triple(scaleC to 0, it, 1) }
                    // softmax over last dim
                    val atS = spec(g, ql, sk)
                    val at = op("softmax", mapOf("axis" to 2), listOf(scdS), atS)
                        .also { edges += Triple(scd to 0, it, 0) }
                    // out chunk = attn @ Vg
                    val ocS = spec(g, ql, d)
                    val oc = op("matmul", emptyMap(), listOf(atS, vgS), ocS)
                        .also { edges += Triple(at to 0, it, 0); edges += Triple(vg to 0, it, 1) }
                    seqOuts += oc
                    qs = qe
                }
                // concat query chunks back to [g,Sq,D] (single chunk => use it directly)
                val og = if (seqOuts.size == 1) {
                    seqOuts[0]
                } else {
                    val ogS = spec(g, sq, d)
                    op("concat", mapOf("dim" to 1), seqOuts.map { it.outputs[0] }, ogS).also { c ->
                        seqOuts.forEachIndexed { i, n -> edges += Triple(n to 0, c, i) }
                    }
                }
                groupOuts += og to g
                s = e
            }

            // concat groups along heads -> [H,Sq,D], then reshape -> [1,H,Sq,D]
            val catS = spec(h, sq, d)
            val cat = op("concat", mapOf("dim" to 0), groupOuts.map { it.first.outputs[0] }, catS)
            groupOuts.forEachIndexed { i, (n, _) -> edges += Triple(n to 0, cat, i) }
            val fin = op("reshape", emptyMap(), listOf(catS), out.copy(name = "${base}_out"))
            edges += Triple(cat to 0, fin, 0)

            // Commit nodes + internal edges.
            for (n in nodes) newGraph.addNode(n)
            for ((src, dst, inIdx) in edges) {
                newGraph.addEdge(
                    GraphEdge("e_${src.first.id}_${src.second}__${dst.id}_$inIdx", src.first, dst, src.second, inIdx, dst.inputs[inIdx]),
                )
            }
            // Rewire the SDPA's consumers to the final reshape.
            for (c in consumersOf[sdpa.id].orEmpty()) {
                newGraph.addEdge(c.copy(id = "${c.id}_torq", source = fin, sourceOutputIndex = 0))
            }
        }

        // Re-add all edges that don't touch an SDPA node.
        for (e in graph.edges) {
            if (e.source.id in sdpaIds || e.destination.id in sdpaIds) continue
            edgeSafeAdd(newGraph, e)
        }
        return GraphOptimizationResult(newGraph, changed = true)
    }

    private fun edgeSafeAdd(g: DefaultComputeGraph, e: GraphEdge) {
        val byId = g.nodes.associateBy { it.id }
        val s = byId[e.source.id] ?: return
        val d = byId[e.destination.id] ?: return
        g.addEdge(e.copy(source = s, destination = d))
    }
}
