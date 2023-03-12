module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<2000x2300xf64>, %arg3: tensor<2000x2600xf64>, %arg4: tensor<2600x2300xf64>) -> tensor<2000x2300xf64> attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg3, %arg4 : (tensor<2000x2600xf64>, tensor<2600x2300xf64>) -> tensor<2000x2300xf64>
    %1 = chlo.broadcast_multiply %0, %arg0 : (tensor<2000x2300xf64>, tensor<f64>) -> tensor<2000x2300xf64>
    %2 = chlo.broadcast_multiply %arg2, %arg1 : (tensor<2000x2300xf64>, tensor<f64>) -> tensor<2000x2300xf64>
    %3 = stablehlo.add %2, %1 : tensor<2000x2300xf64>
    return %3 : tensor<2000x2300xf64>
  }
}
