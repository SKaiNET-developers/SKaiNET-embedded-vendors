# skainet/iree-compiler-torq

The Synaptics Torq-fork IREE compiler as a container, with the subcommand interface of `skainet/iree-compiler`
(SKaiNET-IREE-tools), so cartridge blueprints can switch the compiler image per target.

The SL26xx board runtime is the Torq fork of IREE. It rejects modules compiled by upstream IREE
("required module features [Ch] are not available"), so **every** graph for the board comes from this compiler,
the CPU ones too, at the same Torq version as the board runtime. This image does not inherit from the
SKaiNET-IREE-tools images on purpose (see that repo's `docs/VENDOR_LAYERS.md`).

| Command | Does |
|---|---|
| `compile-cpu <in.mlir> --out f.vmfb [--target linux-arm64\|host] [--cpu c]` | llvm-cpu, embedded ELF; `linux-arm64` = Cortex-A55, v8.2a + dotprod |
| `compile-torq <in.mlir> --out f.vmfb [--target SL2610] [-- extra]` | Torq NPU (`--torq-fallback-f32-to-host`) |
| `compile -- <iree-compile args>` | escape hatch |
| `version`, `shell` | |

Parameter archives (`.irpa`) are not produced here: the wheel has no `iree-convert-parameters`, and the format is
shared, so `skainet/iree-compiler convert-parameters` makes them.

```
synaptics-torq/docker/build.sh                       # pins: torq.env (TORQ_VERSION, TORQ_MANYLINUX, TORQ_BASE_IMAGE)
docker run --rm -v $PWD:/work skainet/iree-compiler-torq:2.3.0 compile-cpu model.mlir --out model.vmfb
```
