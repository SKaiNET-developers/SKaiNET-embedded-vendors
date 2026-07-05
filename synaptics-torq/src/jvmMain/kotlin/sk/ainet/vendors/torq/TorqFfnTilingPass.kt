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
 * Torq-**target** graph-lowering pass — **hidden-dimension tiling of the MLP/FFN**.
 *
 * The Moonshine FFN is a plain `up → gelu → down` MLP whose intermediate is
 * `[·, S, ffnDim]` (e.g. `[1,165,1152]`). On Torq **stable v2.0.0** this overflows
 * LRAM — and, crucially, it overflows because the FFN **weights** dominate
 * (`[dim,ffnDim]` + `[ffnDim,dim]` ≈ 1.3 MB), so sequence tiling does NOT help. The
 * fix is to tile the **hidden (ffnDim) dimension**: split the up-projection weight into
 * column chunks and the down-projection weight into the matching row chunks, so each
 * chunk computes `up_c = A·Wu_c → gelu → p_c = g_c·Wd_c` over a `[·,S,chunkSize]`
 * intermediate, and the partial down-projections are **summed**. The full
 * `[·,S,ffnDim]` intermediate is never materialised.
 *
 * Matches the `matmul → gelu → matmul` triple in the compute graph (GELU is a single
 * node before StableHLO lowering). Output is still portable StableHLO (runs on
 * llvm-cpu too); no HW knowledge leaks into core or the model. Plugs into
 * `TargetOptimizers` for the `"torq"` target alongside [TorqAttentionTilingPass].
 */
class TorqFfnTilingPass(private val hiddenTile: Int = 288) : GraphOptimizationPass {
    override val name: String = "torq-ffn-tiling"

    // A resolved FFN: up-matmul -> (optional +fc1.bias) -> gelu -> down-matmul -> (optional +fc2.bias).
    // The bias adds are OPTIONAL so this tiles both the bias-free MLP and the faithful
    // Moonshine MLP (which carries mlp.fc1.bias / mlp.fc2.bias). They stay in the model as
    // plain `add` ops (HW-agnostic); this Torq plugin folds them into the hidden-tiling.
    private data class Triple3(
        val up: GraphNode,
        val gelu: GraphNode,
        val down: GraphNode,
        val addUp: GraphNode?,
        val addDown: GraphNode?,
    )

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        // Producer of each (nodeId, inputIndex): (source node, source output index).
        val producerOf = HashMap<Pair<String, Int>, Pair<GraphNode, Int>>()
        for (e in graph.edges) producerOf[e.destination.id to e.destinationInputIndex] = e.source to e.sourceOutputIndex
        val consumersOf = graph.edges.groupBy { it.source.id }

        fun isMatmul(n: GraphNode?) = n != null && n.operationName.lowercase() == "matmul"
        fun isAdd(n: GraphNode?) = n != null && n.operationName.lowercase() == "add"

        // Find every up→(+bias)→gelu→down→(+bias) FFN that we can fully resolve.
        val triples = mutableListOf<Triple3>()
        for (gelu in graph.nodes.filter { it.operationName.lowercase() == "gelu" }) {
            val geluIn = producerOf[gelu.id to 0]?.first ?: continue
            // gelu's producer is either the up-matmul directly, or an `add` (up-matmul + fc1.bias).
            val addUp: GraphNode?
            val up: GraphNode?
            if (isMatmul(geluIn)) {
                up = geluIn; addUp = null
            } else if (isAdd(geluIn)) {
                addUp = geluIn; up = producerOf[geluIn.id to 0]?.first
            } else continue
            if (!isMatmul(up)) continue
            val down = consumersOf[gelu.id].orEmpty()
                .firstOrNull { isMatmul(it.destination) && it.destinationInputIndex == 0 }?.destination ?: continue
            // Optional fc2.bias add consuming the down-matmul output at input 0.
            val addDown = consumersOf[down.id].orEmpty()
                .firstOrNull { isAdd(it.destination) && it.destinationInputIndex == 0 }?.destination
            // Need producers for A (up in0), Wu (up in1), Wd (down in1), and each present bias.
            if (producerOf[up!!.id to 0] == null || producerOf[up.id to 1] == null || producerOf[down.id to 1] == null) continue
            if (addUp != null && producerOf[addUp.id to 1] == null) continue
            if (addDown != null && producerOf[addDown.id to 1] == null) continue
            triples += Triple3(up, gelu, down, addUp, addDown)
        }
        if (triples.isEmpty()) return GraphOptimizationResult(graph, changed = false)

        val newGraph = DefaultComputeGraph()
        val removed = triples.flatMap {
            listOfNotNull(it.up.id, it.gelu.id, it.down.id, it.addUp?.id, it.addDown?.id)
        }.toSet()
        for (n in graph.nodes.filter { it.id !in removed }) newGraph.addNode(n)

        for (t in triples) {
            val (up, gelu, down) = t
            val a = up.inputs[0]              // activation  [·, S, dim]
            val wu = up.inputs[1]             // up weight   [dim, ffnDim]
            val wd = down.inputs[1]           // down weight [ffnDim, dim]
            val geluOut = gelu.outputs[0]     // [·, S, ffnDim]
            val downOut = down.outputs[0]     // [·, S, dim]
            val elem = geluOut.dtype
            val ffnDim = geluOut.shape!!.last()
            val wuAxis = wu.shape!!.size - 1  // ffnDim axis of the up weight (N, non-contracted)
            val wdAxis = wd.shape!!.size - 2  // ffnDim axis of the down weight (K, contracted)

            val base = gelu.id
            var counter = 0
            fun spec(dims: List<Int>) = TensorSpec("${base}_ft${counter++}", dims, elem)
            val nodes = mutableListOf<GraphNode>()
            val edges = mutableListOf<kotlin.Triple<Pair<GraphNode, Int>, GraphNode, Int>>()
            fun op(opName: String, params: Map<String, Any>, ins: List<TensorSpec>, o: TensorSpec): GraphNode {
                val n = GraphNode("${base}_${opName}_${counter++}", GenericOperation(opName, params), ins, listOf(o))
                nodes += n; return n
            }

            val aProd = producerOf[up.id to 0]!!
            val wuProd = producerOf[up.id to 1]!!
            val wdProd = producerOf[down.id to 1]!!

            // Even hidden chunks of ≤hiddenTile.
            val nChunk = (ffnDim + hiddenTile - 1) / hiddenTile
            val chunkSize = (ffnDim + nChunk - 1) / nChunk
            val partials = mutableListOf<GraphNode>()
            var cs = 0
            while (cs < ffnDim) {
                val ce = minOf(cs + chunkSize, ffnDim); val cl = ce - cs
                // slice up weight columns -> [dim, cl]
                val wuChunkShape = wu.shape!!.toMutableList().also { it[wuAxis] = cl }
                val wuStart = MutableList(wu.shape!!.size) { 0 }.also { it[wuAxis] = cs }
                val wuLimit = wu.shape!!.toMutableList().also { it[wuAxis] = ce }
                val wuC = op(
                    "slice",
                    mapOf("start_indices" to wuStart, "limit_indices" to wuLimit, "strides" to List(wu.shape!!.size) { 1 }),
                    listOf(wu), spec(wuChunkShape),
                ).also { edges += kotlin.Triple(wuProd, it, 0) }
                // up_c = A @ Wu_c  -> [·, S, cl]
                val upShape = a.shape!!.toMutableList().also { it[it.size - 1] = cl }
                val upC = op("matmul", emptyMap(), listOf(a, wuC.outputs[0]), spec(upShape))
                    .also { edges += kotlin.Triple(aProd, it, 0); edges += kotlin.Triple(wuC to 0, it, 1) }
                // + fc1.bias chunk (faithful MLP): slice the [ffnDim] bias to [cl] and add,
                // broadcasting over the leading dims exactly as the untiled `matmul + bias`.
                val upBiased = if (t.addUp != null) {
                    val bu = t.addUp.inputs[1]
                    val buProd = producerOf[t.addUp.id to 1]!!
                    val buC = op(
                        "slice",
                        mapOf("start_indices" to listOf(cs), "limit_indices" to listOf(ce), "strides" to listOf(1)),
                        listOf(bu), spec(listOf(cl)),
                    ).also { edges += kotlin.Triple(buProd, it, 0) }
                    op("add", emptyMap(), listOf(upC.outputs[0], buC.outputs[0]), spec(upShape))
                        .also { edges += kotlin.Triple(upC to 0, it, 0); edges += kotlin.Triple(buC to 0, it, 1) }
                } else {
                    upC
                }
                // g_c = gelu(up_c [+ bias])
                val gC = op("gelu", emptyMap(), listOf(upBiased.outputs[0]), spec(upShape))
                    .also { edges += kotlin.Triple(upBiased to 0, it, 0) }
                // slice down weight rows -> [cl, dim]
                val wdChunkShape = wd.shape!!.toMutableList().also { it[wdAxis] = cl }
                val wdStart = MutableList(wd.shape!!.size) { 0 }.also { it[wdAxis] = cs }
                val wdLimit = wd.shape!!.toMutableList().also { it[wdAxis] = ce }
                val wdC = op(
                    "slice",
                    mapOf("start_indices" to wdStart, "limit_indices" to wdLimit, "strides" to List(wd.shape!!.size) { 1 }),
                    listOf(wd), spec(wdChunkShape),
                ).also { edges += kotlin.Triple(wdProd, it, 0) }
                // p_c = g_c @ Wd_c  -> [·, S, dim]
                val pC = op("matmul", emptyMap(), listOf(gC.outputs[0], wdC.outputs[0]), spec(downOut.shape!!))
                    .also { edges += kotlin.Triple(gC to 0, it, 0); edges += kotlin.Triple(wdC to 0, it, 1) }
                partials += pC
                cs = ce
            }

            // Sum the partial down-projections (balanced tree of adds).
            var level = partials.toList()
            while (level.size > 1) {
                val next = mutableListOf<GraphNode>()
                var i = 0
                while (i < level.size) {
                    if (i + 1 < level.size) {
                        val add = op("add", emptyMap(), listOf(level[i].outputs[0], level[i + 1].outputs[0]), spec(downOut.shape!!))
                            .also { edges += kotlin.Triple(level[i] to 0, it, 0); edges += kotlin.Triple(level[i + 1] to 0, it, 1) }
                        next += add
                    } else {
                        next += level[i]
                    }
                    i += 2
                }
                level = next
            }
            // Add fc2.bias ONCE to the accumulated partial sum (faithful MLP output).
            val sum = if (t.addDown != null) {
                val bd = t.addDown.inputs[1]
                val bdProd = producerOf[t.addDown.id to 1]!!
                op("add", emptyMap(), listOf(level[0].outputs[0], bd), spec(downOut.shape!!))
                    .also { edges += kotlin.Triple(level[0] to 0, it, 0); edges += kotlin.Triple(bdProd, it, 1) }
            } else {
                level[0]
            }

            for (n in nodes) newGraph.addNode(n)
            for ((src, dst, inIdx) in edges) {
                newGraph.addEdge(
                    GraphEdge("e_${src.first.id}_${src.second}__${dst.id}_$inIdx", src.first, dst, src.second, inIdx, dst.inputs[inIdx]),
                )
            }
            // Rewire the FFN output's consumers (the fc2.bias add if present, else the
            // down-matmul) to our accumulated, bias-added sum.
            val ffnOutId = (t.addDown ?: down).id
            for (c in consumersOf[ffnOutId].orEmpty()) {
                newGraph.addEdge(c.copy(id = "${c.id}_torqffn", source = sum, sourceOutputIndex = 0))
            }
        }

        // Re-add every edge that doesn't touch a removed node.
        for (e in graph.edges) {
            if (e.source.id in removed || e.destination.id in removed) continue
            val byId = newGraph.nodes.associateBy { it.id }
            val s = byId[e.source.id] ?: continue
            val d = byId[e.destination.id] ?: continue
            newGraph.addEdge(e.copy(source = s, destination = d))
        }
        return GraphOptimizationResult(newGraph, changed = true)
    }
}
