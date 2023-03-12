module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<2800x2800xf64>, %arg3: tensor<2800x2800xf64>, %arg4: tensor<2800xf64>, %arg5: tensor<2800xf64>, %arg6: tensor<2800xf64>) -> tensor<2800xf64> attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg2, %arg5 : (tensor<2800x2800xf64>, tensor<2800xf64>) -> tensor<2800xf64>
    %1 = chlo.broadcast_multiply %0, %arg0 : (tensor<2800xf64>, tensor<f64>) -> tensor<2800xf64>
    %2 = stablehlo.dot %arg3, %arg5 : (tensor<2800x2800xf64>, tensor<2800xf64>) -> tensor<2800xf64>
    %3 = chlo.broadcast_multiply %2, %arg1 : (tensor<2800xf64>, tensor<f64>) -> tensor<2800xf64>
    %4 = stablehlo.add %1, %3 : tensor<2800xf64>
    return %4 : tensor<2800xf64>
  }
}
