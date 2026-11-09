# skainet/iree-runtime-torq

The Torq-fork IREE runtime for SL26xx boards (aarch64), source-built from the public
[torq-compiler](https://github.com/synaptics-torq/torq-compiler) tag and the IREE fork commit that tag pins. It is the
counterpart of `skainet/iree-compiler-torq`: a module only loads on a runtime of the same Torq version, so both take
their version from `synaptics-torq/docker/torq.env`. Build it locally:

```sh
synaptics-torq/docker-runtime/build.sh
docker run --rm -v "$PWD/dist:/out" skainet/iree-runtime-torq:2.3.0 dist /out
```

| Command | Does |
|---|---|
| `dist <out dir>` | copies `lib/libiree_torq_runtime.a` (local-sync, local-task, vulkan, torq), `include/`, `licenses/` and `BUILD-INFO` |
| `version`, `shell` | |

The archive uses the Kotlin/Native clang-21 and gcc-8.3/glibc-2.25 sysroot (`KONAN_LLVM`, `KONAN_GCC` in the image), so
it links into a Kotlin/Native linuxArm64 binary. Consumers define `-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl`
and link `-lstdc++ -lstdc++fs -lm -ldl -lpthread`. Do not use `--whole-archive`: the archive holds duplicate objects.

`compat-include/` holds the dma-buf and dma-heap uapi headers (`GPL-2.0 WITH Linux-syscall-note`), which the sysroot's
kernel 4.9 headers lack.
