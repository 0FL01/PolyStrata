# Inkling-2-Mini CPU image encoder

Build the resident `strata-vision` helper with the `inkling2v` implementation from
[llama.cpp #32695](https://github.com/ggml-org/llama.cpp/pull/32695). Use a separate checkout and build
directory. The helper opens the text GGUF for its vocabulary only; `strata-poly` still runs the text model.

From the PolyStrata root on Linux, with `work` set to a new directory outside this repo:

```sh
pin=6196d9802e0757a5218f16f91d9a6830793ba800
git init "$work/llama.cpp"
git -C "$work/llama.cpp" fetch --depth=1 https://github.com/ggml-org/llama.cpp "$pin"
git -C "$work/llama.cpp" checkout --detach "$pin"
cmake -S tools/vision -B "$work/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$work/llama.cpp" -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_CUDA=OFF \
  -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF
cmake --build "$work/build" --target strata-vision -j4
"$work/build/bin/strata-vision" --mmproj "$model_dir/mmproj-F16.gguf" --model "$text_gguf" --threads 4 --max-tokens 1024
```

Set `model_dir` to the directory with the Inkling projector, and `text_gguf` to the Inkling-2-Mini text GGUF.
After `READY 2304`, send `ENC /absolute/image.png /absolute/rows.sve` on stdin. The same process accepts
another `ENC` without reloading the projector. `QUIT` ends it. The native build targets the CPU used to build it.

To serve images, add a `vision` section to the Inkling `strata-poly` run config (replace the paths):

```json
"vision": {
  "exe": "/path/to/the/inkling-build/bin/strata-vision",
  "model": "/path/to/Inkling-2-Mini-UD-IQ4_XS.gguf",
  "mmproj": "/path/to/Inkling-2-Mini/mmproj-F16.gguf",
  "gpu": false,
  "elastic": false,
  "threads": 4,
  "max_tokens": 1024
}
```

`elastic: false` keeps the CPU encoder in system RAM between requests. Do not add `--vision` to the
`strata-poly` arguments. Without this section the server remains text-only. With it, `/v1/models`
lists `image` among the model's input modalities; send pictures as ordinary `image_url` content parts.

On a Ryzen 7 5700X with four encoder threads and `--max-tokens 1024`, the F16 projector produced 4 rows for
an 80×80 PNG, 15 for 161×87 and 256 for 640×640. Every row contained 2304 finite floats. Inkling uses sequential
decoder positions, so the SVE1 grid is `n_tokens × 1` even for a square picture. The same helper process used
701 MiB RSS at `READY` and 743 MiB after the 640×640 encode, with no GPU device open.
