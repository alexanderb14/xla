module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<2000x2000xf32>, %arg3: tensor<2000xf32>, %arg4: tensor<2000xf32>, %arg5: tensor<2000xf32>, %arg6: tensor<2000xf32>, %arg7: tensor<2000xf32>, %arg8: tensor<2000xf32>, %arg9: tensor<2000xf32>, %arg10: tensor<2000xf32>) -> tensor<2000xf32> attributes {irsynth.raised} {
    %0 = stablehlo.reshape %arg3 {new_sizes = dense<[3, 1]> : tensor<2xi64>} : (tensor<2000xf32>) -> tensor<2000x1xf32>
    %1 = stablehlo.reshape %arg4 {new_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<2000xf32>) -> tensor<1x2000xf32>
    %2 = stablehlo.dot %0, %1 : (tensor<2000x1xf32>, tensor<1x2000xf32>) -> tensor<2000x2000xf32>
    %3 = stablehlo.reshape %arg5 {new_sizes = dense<[3, 1]> : tensor<2xi64>} : (tensor<2000xf32>) -> tensor<2000x1xf32>
    %4 = stablehlo.reshape %arg6 {new_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<2000xf32>) -> tensor<1x2000xf32>
    %5 = stablehlo.dot %3, %4 : (tensor<2000x1xf32>, tensor<1x2000xf32>) -> tensor<2000x2000xf32>
    %6 = stablehlo.add %arg2, %2 : tensor<2000x2000xf32>
    %7 = stablehlo.add %6, %5 : tensor<2000x2000xf32>
    %8 = chlo.broadcast_multiply %7, %arg1 : (tensor<2000x2000xf32>, tensor<f32>) -> tensor<2000x2000xf32>
    %9 = stablehlo.dot %arg9, %8 : (tensor<2000xf32>, tensor<2000x2000xf32>) -> tensor<2000xf32>
    %10 = stablehlo.add %arg8, %9 : tensor<2000xf32>
    %11 = stablehlo.add %10, %arg10 : tensor<2000xf32>
    %12 = chlo.broadcast_multiply %7, %arg0 : (tensor<2000x2000xf32>, tensor<f32>) -> tensor<2000x2000xf32>
    %13 = stablehlo.dot %12, %11 : (tensor<2000x2000xf32>, tensor<2000xf32>) -> tensor<2000xf32>
    %14 = stablehlo.add %arg7, %13 : tensor<2000xf32>
    return %14 : tensor<2000xf32>
  }
}

