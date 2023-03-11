func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<1600x1800xf64>, %arg3: tensor<1600x2200xf64>, %arg4: tensor<2200x1800xf64>, %arg5: tensor<1800x2400xf64>, %arg6: tensor<1600x2400xf64>) -> tensor<1600x2400xf64> {
  %0 = chlo.broadcast_multiply %arg4, %arg0 : (tensor<2200x1800xf64>, tensor<f64>) -> tensor<2200x1800xf64>
  %1 = stablehlo.dot %arg3, %0 : (tensor<1600x2200xf64>, tensor<2200x1800xf64>) -> tensor<1600x1800xf64>
  %2 = stablehlo.dot %1, %arg5 : (tensor<1600x1800xf64>, tensor<1800x2400xf64>) -> tensor<1600x2400xf64>
  %3 = chlo.broadcast_multiply %arg6, %arg1 : (tensor<1600x2400xf64>, tensor<f64>) -> tensor<1600x2400xf64>
  %4 = stablehlo.add %3, %2 : tensor<1600x2400xf64>
  return %4 : tensor<1600x2400xf64>
}
