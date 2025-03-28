load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "mhlo_to_lhlo_with_xla",
    srcs = ["mhlo_to_lhlo_with_xla.cc"],
    hdrs = ["mhlo_to_lhlo_with_xla.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/service:backend",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_parser",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/llvm_ir:buffer_assignment_util",
        "//xla/translate/hlo_to_mhlo:attribute_importer",
        "//xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TranslateLib",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)

xla_cc_binary(
    name = "xla-translate-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/framework/transforms:passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:hlo_xla_runtime_pipeline",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "xla-translate-gpu-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/framework/transforms:passes",
        "//xla/mlir_hlo:all_passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/service:gpu_plugin",
        "//xla/stream_executor:stream_executor_impl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
        "@tsl//tsl/platform:platform_port",
    ],
)
