load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_py_proto_library",
    "xla_py_test_deps",
)
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "polybench_lib",
    srcs = ["polybench.cc"],
    hdrs = ["polybench.h",
            "atax.h",
            "2mm.h",
            "3mm.h",
            "bicg.h",
            "gemm.h",
            "gemver.h",
            "gesummv.h",
            "mvt.h",
            "syrk.h",
            "syr2k.h",
            "doitgen.h",
            "covariance.h",
            "correlation.h",
            "symm.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "xla_compiler_lib",
    srcs = ["xla_compiler.cc"],
    hdrs = ["xla_compiler.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:literal_util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/pjrt:local_device_state",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_binary(
    name = "atax",
    srcs = ["atax.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "2mm",
    srcs = ["2mm.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "3mm",
    srcs = ["3mm.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "bicg",
    srcs = ["bicg.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "gemm",
    srcs = ["gemm.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "gemver",
    srcs = ["gemver.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "gesummv",
    srcs = ["gesummv.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "mvt",
    srcs = ["mvt.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "syrk",
    srcs = ["syrk.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "syr2k",
    srcs = ["syr2k.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "doitgen",
    srcs = ["doitgen.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "covariance",
    srcs = ["covariance.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "correlation",
    srcs = ["correlation.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "symm",
    srcs = ["symm.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)

xla_cc_binary(
    name = "xla_runner",
    srcs = ["xla_runner.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
        ":xla_compiler_lib",
    ],
)
