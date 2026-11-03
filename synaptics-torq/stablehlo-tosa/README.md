# skainet/stablehlo-tosa

StableHLO's own TOSA legalization as a container: `stablehlo-opt --stablehlo-prepare-for-tosa
--stablehlo-legalize-to-tosa`. SKaiNET exports StableHLO; Synaptics' Torq compiler makes a function **one NPU
dispatch** only on its TOSA (and ONNX/Torch) input pipelines, so a SKaiNET graph goes StableHLO → TOSA → `compile-torq
--input tosa-torq` (`../docker`). The Torq wheel ships no `stablehlo-opt`.

It is built from the Torq compiler's sources (`synaptics-torq/torq-compiler` at `v<TORQ_VERSION>`, stock IREE CMake, no
Torq plugin, target `stablehlo-opt` only), not from standalone StableHLO: that one pins its own LLVM, and the TOSA text
must parse in the Torq-fork `iree-compile`, which uses `synaptics-torq/llvm-project`. Keep `TORQ_VERSION` equal to
`skainet/iree-compiler-torq`'s.

```
synaptics-torq/stablehlo-tosa/build.sh          # pins: ../docker/torq.env; ~35 min on 12 cores, builder image pinned by digest
docker run --rm -v $PWD:/work skainet/stablehlo-tosa:2.3.0 model.mlir --stablehlo-prepare-for-tosa --stablehlo-legalize-to-tosa -o model.tosa.mlir
```

The legalization does not cover every StableHLO op. On SKaiNET's Moonshine v2 encoder it leaves `broadcast_in_dim`,
`sqrt` and most `dot_general` forms, which SKaiNET-cartridge-blueprints' `torq-tosa` step translates (and it rewrites
divides by a broadcast before the legalization). `build-stablehlo-opt.sh` is the build itself, also usable in the
builder image on an existing checkout.
