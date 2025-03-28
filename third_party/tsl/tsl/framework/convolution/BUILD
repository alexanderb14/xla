load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    features = [
        # Required since headers are not self-contained.
        "-parse_headers",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "eigen_spatial_convolutions-inl",
    hdrs = [
        "eigen_spatial_convolutions-inl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    # Hack to disable breaking AVX512 special GemmKernel. There is a conflicting
    # specialization there causing build breakages.  This must be added here
    # as "defines" so that the header is excluded in all dependent targets.
    # TODO(b/238649163): remove this once no longer necessary.
    defines = ["GEMM_KERNEL_H"],
    deps = [
        "//tsl/framework/convolution:eigen_convolution_helpers",
    ],
)

cc_library(
    name = "eigen_convolution_helpers",
    hdrs = [
        "eigen_convolution_helpers.h",
    ],
    compatible_with = get_compatible_with_portable(),
    defines = [
        "EIGEN_ALTIVEC_USE_CUSTOM_PACK=0",
    ],
)

# Tensorflow also has an eigen_helpers that is closely related, so maintain the same name.
cc_library(
    name = "eigen_helpers",
    hdrs = [
        "eigen_spatial_convolutions.h",
    ],
    compatible_with = get_compatible_with_portable(),
    defines = ["EIGEN_NEON_GEBP_NR=4"],
    deps = [
        "//tsl/framework/contraction:eigen_contraction_kernel",
        "//tsl/framework/convolution:eigen_convolution_helpers",
        "//tsl/framework/convolution:eigen_spatial_convolutions-inl",
        "@eigen_archive//:eigen3",
    ],
)

# Maintain the same name as other directories until a principled refactor is done, as these files
# used to all be a single target.
filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "eigen_convolution_helpers.h",
        "eigen_spatial_convolutions.h",
        "eigen_spatial_convolutions-inl.h",
    ],
    # Somehow the following code works with fixedpoint, but not here.
    # visibility = [
    #     "//tensorflow:__subpackages__",
    #     "//tsl:internal",
    # ],
)

# Test against platforms and architecures that don't support CUDA.
# TODO(b/153737462): Automatically filter tests to create the appropriate
# portable test list.
test_suite(
    name = "portable_kernel_tests",
    tags = [
        "manual",  # Avoid redundancy when using wildcard test patterns.
    ],
    tests = [
        ":eigen_spatial_convolutions_test",
    ],
)

# Portable Tensorflow for Android/iOS requires these files directly rather than as libraries, so
# export them to be used there.
exports_files(
    srcs = [
        "eigen_convolution_helpers.h",
        "eigen_spatial_convolutions.h",
        "eigen_spatial_convolutions-inl.h",
    ],
)

tsl_cc_test(
    name = "spatial_convolutions_test",
    size = "small",
    srcs = [
        "eigen_spatial_convolutions_test.cc",
    ],
    deps = [
        ":eigen_helpers",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)
