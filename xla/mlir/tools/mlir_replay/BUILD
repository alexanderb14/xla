load("//xla:xla.bzl", "xla_cc_binary")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

xla_cc_binary(
    name = "mlir_replay",
    srcs = ["mlir_replay.cc"],
    deps = [
        ":mlir_replay_lib",
        "//xla:debug_options_flags",
        "//xla/mlir/runtime/ir:rt",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc_impl",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:gml_st",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/mlir_hlo:mlir_interpreter_dialects",
        "//xla/mlir_hlo:mlir_interpreter_framework",
        "//xla/mlir_hlo:thlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "mlir_replay_lib",
    srcs = ["mlir_replay_lib.cc"],
    hdrs = ["mlir_replay_lib.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_proto_cc",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//xla/mlir_hlo:mlir_interpreter_framework",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)
