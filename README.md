# OpenXLA

OpenXLA is a community-driven, modular open-source compiler. The project is
actively migrating from
[tensorflow/xla](https://github.com/tensorflow/tensorflow/tree/e2009cbe954b5c7644eecd77243cd4dfee14ff8d/tensorflow/compiler/xla)
under the governance of the
[OpenXLA SIG](https://github.com/tensorflow/community/pull/419/).

> **Note** During the migration, the project cannot accept pull requests.

The OpenXLA compiler will enable efficient optimization and deployment of ML
models from most major frameworks to any hardware backend, including CPUs, GPUs,
and ML ASICs. To learn more, see [the documentation](docs/index.md).

## Get started

If you want to use XLA to compile your ML project, refer to the corresponding
documentation for your ML framework:

* [PyTorch](https://pytorch.org/xla)
* [TensorFlow](https://www.tensorflow.org/xla)
* [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

If you're not contributing code to the XLA compiler, you don't need to clone and
build this repo. Everything here is intended for XLA contributors who want to
develop the compler and XLA integrators who want to debug or add support for ML
frontends and hardware backends.

Here's how you can start developing in the XLA compiler:

To build XLA, you will need to install [Bazel](https://bazel.build/install).
[Bazelisk](https://github.com/bazelbuild/bazelisk#readme) is an easy way to
install Bazel and automatically downloads the correct Bazel version for XLA. If
Bazelisk is unavailable, you can manually install Bazel instead.

Clone this repository:

```sh
git clone https://github.com/openxla/xla && cd xla
```

We recommend using a suitable docker container to build/test XLA, such as
[TensorFlow's docker container](https://www.tensorflow.org/install/docker):

```
docker run --name xla -w /xla -it -d --rm -v $PWD:/xla tensorflow/build:latest-python3.9 bash
```

Run an end to end test using an example StableHLO module:

```
docker exec xla bazel test xla/examples/axpy:stablehlo_compile_test --nocheck_visibility --test_output=all
```

This will take quite a while your first time because it must build the entire
stack, including MLIR, StableHLO, XLA, and more.

When it's done, you should see output like this:

```sh
==================== Test output for //xla/examples/axpy:stablehlo_compile_test:
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from StableHloAxpyTest
[ RUN      ] StableHloAxpyTest.LoadAndRunCpuExecutable
Loaded StableHLO program from xla/examples/axpy/stablehlo_axpy.mlir:
func.func @main(
  %alpha: tensor<f32>, %x: tensor<4xf32>, %y: tensor<4xf32>
) -> tensor<4xf32> {
  %0 = stablehlo.broadcast_in_dim %alpha, dims = []
    : (tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.multiply %0, %x : tensor<4xf32>
  %2 = stablehlo.add %1, %y : tensor<4xf32>
  func.return %2: tensor<4xf32>
}

Computation inputs:
        alpha:f32[] 3.14
        x:f32[4] {1, 2, 3, 4}
        y:f32[4] {10.5, 20.5, 30.5, 40.5}
Computation output: f32[4] {13.64, 26.78, 39.920002, 53.06}
[       OK ] StableHloAxpyTest.LoadAndRunCpuExecutable (264 ms)
[----------] 1 test from StableHloAxpyTest (264 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (264 ms total)
[  PASSED  ] 1 test.
```

## Contacts

*   For questions, contact Thea Lamkin - thealamkin at google.com.

## Resources

*   GitHub
    ([current](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla))
*   [Community Resources](https://github.com/openxla/community)

## Code of Conduct

While under TensorFlow governance, all community spaces for SIG OpenXLA are
subject to the
[TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).

