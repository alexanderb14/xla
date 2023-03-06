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
            "doitgen.h"],
    visibility = ["//visibility:public"],
)

xla_cc_binary(
    name = "stablehlo_compile_test",
    srcs = ["stablehlo_compile_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":polybench_lib",
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
