func.func @main(%arg0: tensor<2200x1800xf64>, %arg1: tensor<1800xf64>, %arg2: tensor<2200xf64>, %arg3: tensor<1800xf64>, %arg4: tensor<2200xf64>) -> (tensor<1800xf64>, tensor<2200xf64>) attributes {irsynth.raised} {
  %0 = stablehlo.dot %arg4, %arg0 : (tensor<2200xf64>, tensor<2200x1800xf64>) -> tensor<1800xf64>
  %1 = stablehlo.dot %arg0, %arg3 : (tensor<2200x1800xf64>, tensor<1800xf64>) -> tensor<2200xf64>
  return %0, %1 : tensor<1800xf64>, tensor<2200xf64>
}
