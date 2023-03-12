module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<4000x4000xf64>, %arg3: tensor<4000xf64>, %arg4: tensor<4000xf64>, %arg5: tensor<4000xf64>, %arg6: tensor<4000xf64>, %arg7: tensor<4000xf64>, %arg8: tensor<4000xf64>, %arg9: tensor<4000xf64>, %arg10: tensor<4000xf64>) -> tensor<4000xf64> attributes {irsynth.raised} {
    %0 = stablehlo.reshape %arg3 {new_sizes = dense<[3, 1]> : tensor<2xi64>} : (tensor<4000xf64>) -> tensor<4000x1xf64>
    %1 = stablehlo.reshape %arg4 {new_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<4000xf64>) -> tensor<1x4000xf64>
    %2 = stablehlo.dot %0, %1 : (tensor<4000x1xf64>, tensor<1x4000xf64>) -> tensor<4000x4000xf64>
    %3 = stablehlo.reshape %arg5 {new_sizes = dense<[3, 1]> : tensor<2xi64>} : (tensor<4000xf64>) -> tensor<4000x1xf64>
    %4 = stablehlo.reshape %arg6 {new_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<4000xf64>) -> tensor<1x4000xf64>
    %5 = stablehlo.dot %3, %4 : (tensor<4000x1xf64>, tensor<1x4000xf64>) -> tensor<4000x4000xf64>
    %6 = stablehlo.add %arg2, %2 : tensor<4000x4000xf64>
    %7 = stablehlo.add %6, %5 : tensor<4000x4000xf64>
    %8 = chlo.broadcast_multiply %7, %arg1 : (tensor<4000x4000xf64>, tensor<f64>) -> tensor<4000x4000xf64>
    %9 = stablehlo.dot %arg9, %8 : (tensor<4000xf64>, tensor<4000x4000xf64>) -> tensor<4000xf64>
    %10 = stablehlo.add %arg8, %9 : tensor<4000xf64>
    %11 = stablehlo.add %10, %arg10 : tensor<4000xf64>
    %12 = chlo.broadcast_multiply %7, %arg0 : (tensor<4000x4000xf64>, tensor<f64>) -> tensor<4000x4000xf64>
    %13 = stablehlo.dot %12, %11 : (tensor<4000x4000xf64>, tensor<4000xf64>) -> tensor<4000xf64>
    %14 = stablehlo.add %arg7, %13 : tensor<4000xf64>
    return %14 : tensor<4000xf64>
  }
}

