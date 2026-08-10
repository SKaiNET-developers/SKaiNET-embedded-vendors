# IREE toolchain alignment

Shared, hardware-agnostic dockerized IREE tooling lives in
**SKaiNET-iree-toolchain** (`skainet/iree-compiler`, `skainet/iree-android`,
`skainet/iree-dev` — pinned via its `versions.env`). Its
`docs/VENDOR_LAYERS.md` defines the contract vendor images may build on.

This repo is the designated home for **vendor-specific** docker tooling that
cannot inherit those bases. Concretely for `synaptics-torq/`: the Torq
toolchain ships a *forked* IREE compiler (`torq-compiler` wheel), and its
vmfbs are intentionally incompatible with stock IREE — so Torq images do not
`FROM` the shared bases. Alignment is by convention instead: the
`torq-toolchain.lock` mechanism (`COMPILER_ID`, `BOARD_RUNTIME_VERSION`)
pins the Torq toolchain the way `versions.env` pins the stock one. Any future
Torq docker tooling (compiler images, board-runtime packaging) belongs under
`synaptics-torq/` here.
