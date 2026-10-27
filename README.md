# Jaguar

Official implementation of **"Jaguar: Fast Private Neural Network Inference with Power-of-Two Homomorphic Arithmetic,"** accepted to the **NeurIPS 2026 main track**.

Jaguar is a hybrid HE/2PC framework for private neural network inference using power-of-two homomorphic arithmetic. This repository provides end-to-end CNN implementations for ResNet-18, ResNet-50, and MobileNetV2.

## Paper evaluation and repository defaults

**The paper's CNN evaluation uses Cheetah's approximate ReLU protocol with `Q=54`. This repository defaults to Cheetah's exact ReLU protocol with `Q=52`, selected to meet the parameter constraints of the exact-ReLU integration.**

Here, the ciphertext modulus is `q=2^Q`. The current default configuration is:

| Parameter | Default |
| --- | --- |
| Polynomial degree `N` | 2048 |
| Ciphertext modulus `q` | `2^52` |
| Plaintext modulus `p` | `2^30` |
| Activation/weight fractional precision `f` | 9 bits |
| ReLU backend | Exact |
| Batch normalization | Fused into convolution by the run scripts |

The modulus defaults are set in [Jaguar_src/jaguar_main.cpp](Jaguar_src/jaguar_main.cpp), and the ReLU backend is selected in [Jaguar_src/jaguar_relu.cpp](Jaguar_src/jaguar_relu.cpp). The exact backend consumes 21-bit additive shares after truncation and returns 30-bit shares for the next HE layer.

The commands below run the repository's **exact-ReLU configuration**. Its latency and communication should not be treated as a direct reproduction of the paper's approximate-ReLU results. Changing `Q` alone does not restore the paper's evaluation configuration.

## Included inputs and model parameters

**The input `.inp` files and model parameters are already included. No input generation, checkpoint download, or parameter export is required for the examples below.**

| Model argument | Input file | Model parameter files |
| --- | --- | --- |
| `resnet18` | `pretrained/resnet18_input_scale9_pred.inp` | `pretrained/resnet18_model_scale9.inp` and `models/filters_resnet18_224_scale9/layer_*.txt` |
| `resnet50` | `pretrained/resnet50_input_scale9_pred249.inp` | `pretrained/resnet50_model_scale9.inp` |
| `mobilenet_v2` | `pretrained/mobilenet_v2_input_scale9_248.inp` | `models/filters_mobilenetv2_224_scale9/` |

- ResNet-18 uses the layer filter files together with the model `.inp` containing batch-normalization and classifier parameters.
- ResNet-50 loads convolution, batch-normalization, and classifier parameters from its model `.inp`.
- MobileNetV2 uses `layer_*.txt`, `layer_*_bn_scale.txt`, `layer_*_bn_bias.txt`, `fc_weight.txt`, and `fc_bias.txt` in its model directory; it does not require a separate model `.inp`.
- Preserve the supplied filenames and directory structure. Input paths are defined in [networks/input_loader.cpp](networks/input_loader.cpp), and model loading is implemented in `networks/*_support.cpp` and `networks/*_quant.cpp`.
- The model files use the fixed-point formats expected by these loaders. In particular, batch-normalization multipliers use a separate scale handled by the loaders; the `scale9` filenames do not mean that every stored field has the same scale.
- HE configuration is defined in the source code. Runtime key and parameter files are initialized by the executable in the party working directories; users do not need to prepare these files manually.

## Repository layout

| Directory | Contents |
| --- | --- |
| `Jaguar_src/` | HE arithmetic, runtime, transport, activation integration, and profiling |
| `networks/` | Model drivers, input loaders, and parameter loaders |
| `pretrained/` | Prepared input tensors and ResNet model `.inp` files |
| `models/` | Network layer configurations and prepared per-layer parameter files |
| `exact/` | Prebuilt exact-ReLU backend, `libjaguar_exact_v51.so` |
| `SCI/`, `include/gemini/` | Imported/adapted SCI and Cheetah components |
| `deps/` | Vendored build dependencies |
| `scripts/` | Build, run, and result-summary utilities |
| `credits/` | Third-party license copies and attribution information |
| `logs/pair_runs/` | Generated run logs, exit statuses, and summaries |
| `run_client/`, `run_cloud/` | Generated per-party working files |

## Requirements

- Linux on x86-64.
- Bash, Make, CMake 3.x version 3.16 or later, and a C++17 compiler; GCC 11 is recommended.
- OpenSSL development headers/libraries and an OpenMP-capable compiler.
- A CPU supporting the instructions enabled by the build, including AVX2, AES-NI, RDSEED, and SSE4.1. `USE_AVX512=OFF` disables Jaguar's AVX-512 kernels; it does not produce a build without AVX/AVX2.
- Python 3 for result summaries. The summary script uses only the Python standard library; PyTorch, torchvision, NumPy, and Pillow are not required to run the supplied examples.

When an older system compiler is detected, the build script attempts to use `devtoolset-11` through `scl` if available.

### Exact-ReLU runtime compatibility

The default backend is supplied as a prebuilt Linux x86-64 shared library:

```text
exact/libjaguar_exact_v51.so
```

It requires compatible runtime libraries, including `libemp-tool.so`, `libssl.so.10`, and `libcrypto.so.10`. A system providing only OpenSSL 3 libraries will not satisfy these binary dependencies.

The run scripts set `JAGUAR_EXACT_LIBRARY` and the dependency library search path automatically. An alternative compatible backend can be selected through `JAGUAR_EXACT_LIBRARY`. The current build scripts build Jaguar and its vendored dependencies, but do **not** rebuild this exact-ReLU library; its source and build recipe are not included in this snapshot.

## Build

Run all commands below from the repository root.

For a build with Jaguar's AVX-512 kernels disabled:

```bash
env BUILD_DIR=./build_avxoff USE_AVX512=OFF bash scripts/build.sh
```

On a CPU supporting AVX-512, use:

```bash
env BUILD_DIR=./build_avxon USE_AVX512=ON bash scripts/build.sh
```

The build installs dependencies under `${BUILD_DIR}/deps-prefix` and produces `${BUILD_DIR}/jaguar_main`. Dependencies are built from the sources in `deps/`; the prepared model/input files are used as supplied.

To rebuild only Jaguar after its dependencies have been installed:

```bash
cmake --build ./build_avxoff --target client -j
```

Use `./build_avxon` for the AVX-512 build. The CMake target is named `client`, although the output executable is `jaguar_main`.

## Run inference

The pair runner launches both parties on the same machine over loopback:

```bash
env BUILD_DIR=./build_avxoff bash scripts/jaguar_run_model_pair.sh \
  resnet18 demo_resnet18 19001 mt
```

The arguments are `MODEL RUN_ID PORT MODE`:

| Argument | Values |
| --- | --- |
| `MODEL` | `resnet18`, `resnet50`, or `mobilenet_v2` |
| `RUN_ID` | A new name for the output directory under `logs/pair_runs/` |
| `PORT` | Base TCP port; use distinct, available port ranges for separate runs |
| `MODE` | `st` for single-threaded or `mt` for multi-threaded execution |

The exact-ReLU backend also uses `PORT+80` and `PORT+81`. The pair runner uses `127.0.0.1`; it does not configure the LAN/WAN conditions used in the paper.

**The launcher returns after starting the background processes. Wait for both parties to finish before summarizing.** For the example above:

```bash
run_dir=./logs/pair_runs/demo_resnet18
while [ ! -f "$run_dir/client.status" ] || [ ! -f "$run_dir/cloud.status" ]; do
  sleep 1
done
cat "$run_dir/client.status" "$run_dir/cloud.status"
```

Both status files should contain `0`. If either status is nonzero, inspect `client.log` and `cloud.log` in the run directory. If the run stalls, inspect those logs before continuing to wait.

Example commands:

```bash
env BUILD_DIR=./build_avxoff bash scripts/jaguar_run_model_pair.sh \
  resnet18 demo_resnet18 19001 mt

env BUILD_DIR=./build_avxoff bash scripts/jaguar_run_model_pair.sh \
  resnet50 demo_resnet50 19401 mt

env BUILD_DIR=./build_avxoff bash scripts/jaguar_run_model_pair.sh \
  mobilenet_v2 demo_mobilenet_v2 19801 mt
```

For measurements, run one model at a time and wait for it to complete before launching another. Use `BUILD_DIR=./build_avxon` when running the AVX-512 build. Keep `BUILD_DIR` consistent with the build you selected.

## Summarize results

After both parties have completed successfully, run the summary script on the corresponding run directory.

Example commands:

```bash
python3 scripts/summarize_pair_run.py ./logs/pair_runs/demo_resnet18
python3 scripts/summarize_pair_run.py ./logs/pair_runs/demo_resnet50
python3 scripts/summarize_pair_run.py ./logs/pair_runs/demo_mobilenet_v2
```

The summary directory contains:

- `summary/summary.json`: aggregated profiling metrics and communication counters.
- `summary/kind_summary.csv`: breakdown by operation category.
- `summary/kind_summary_client_only.csv`: client-side category breakdown.

In `summary.json`, `structured_total_e2e_time_us` reports **aggregate profiler latency in microseconds**, while `structured_total_comm_bytes_party_sum` reports the **sum of the client and server/cloud communication counters in bytes**. Both totals follow the summary script's filtering rules, with communication summed from both parties' reported `comm_bytes` values. For **wall-clock inference latency**, use the `J52_E2E_US` entry in each party log, which measures the inference region after exact-ReLU and pooling session initialization.

## Citation

If you use Jaguar in your research, please cite:

```bibtex
@inproceedings{jeong2026jaguar,
  title     = {{Jaguar}: Fast Private Neural Network Inference with
               Power-of-Two Homomorphic Arithmetic},
  author    = {Jeong, Yewon and Jung, Nayoung and Roh, Hyeri and Choi, Woo-Seok},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Credits and licensing

Jaguar was developed as part of the research by Yewon Jeong, Nayoung Jung, Hyeri Roh, and Woo-Seok Choi at Seoul National University. Yewon Jeong and Nayoung Jung contributed equally to the paper.

This implementation builds on [OpenCheetah](https://github.com/Alibaba-Gemini-Lab/OpenCheetah) and the [SCI library in EzPC](https://github.com/mpc-msri/EzPC). We thank their authors and the developers of Microsoft SEAL, EMP-toolkit, Eigen, Intel HEXL, Google cpu_features, and the other incorporated components.

Third-party code retains its original copyright notices and applicable license terms. See [credits/](credits/), the license/notice files in `deps/`, and notices embedded in the imported source files. This product includes software developed by the OpenSSL Project for use in the OpenSSL Toolkit (http://www.openssl.org/).

Jaguar's original contributions are licensed under the MIT License; see [LICENSE](LICENSE). Third-party components and bundled model/input assets remain subject to their respective applicable terms. The MIT License for Jaguar's original contributions does not replace or override those terms.
