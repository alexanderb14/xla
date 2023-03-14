func.func @main(%arg0: tensor<1800x2200xf64>, %arg1: tensor<2200xf64>) -> tensor<2200xf64> attributes {irsynth.raised} {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<1800x2200xf64>, tensor<2200xf64>) -> tensor<1800xf64>
  %1 = stablehlo.dot %0, %arg0 : (tensor<1800xf64>, tensor<1800x2200xf64>) -> tensor<2200xf64>
  return %1 : tensor<2200xf64>
}
