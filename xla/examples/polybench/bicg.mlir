func.func @main(%arg0: tensor<2100x1900xf32>, %arg1: tensor<1900xf32>, %arg2: tensor<2100xf32>, %arg3: tensor<1900xf32>, %arg4: tensor<2100xf32>) -> (tensor<1900xf32>, tensor<2100xf32>) attributes {irsynth.raised} {
  %0 = stablehlo.dot %arg4, %arg0 : (tensor<2100xf32>, tensor<2100x1900xf32>) -> tensor<1900xf32>
  %1 = stablehlo.dot %arg0, %arg3 : (tensor<2100x1900xf32>, tensor<1900xf32>) -> tensor<2100xf32>
  return %0, %1 : tensor<1900xf32>, tensor<2100xf32>
}
