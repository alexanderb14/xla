func.func @main(%arg0: tensor<1900x2100xf32>, %arg1: tensor<2100xf32>) -> tensor<2100xf32> attributes {irsynth.raised} {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<1900x2100xf32>, tensor<2100xf32>) -> tensor<1900xf32>
  %1 = stablehlo.dot %0, %arg0 : (tensor<1900xf32>, tensor<1900x2100xf32>) -> tensor<2100xf32>
  return %1 : tensor<2100xf32>
}
