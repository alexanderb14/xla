func.func @main(%arg0: tensor<1800x2200xf64>, %arg1: tensor<2200xf64>) -> tensor<2200xf64> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1800x2200xf64>) -> tensor<2200x1800xf64>
  %1 = stablehlo.dot %arg0, %arg1 : (tensor<1800x2200xf64>, tensor<2200xf64>) -> tensor<1800xf64>
  %2 = stablehlo.dot %0, %1 : (tensor<2200x1800xf64>, tensor<1800xf64>) -> tensor<2200xf64>
  return %2 : tensor<2200xf64>
}
