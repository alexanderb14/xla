module {
  func.func @main(%arg0: tensor<2000xf32>, %arg1: tensor<2000xf32>, %arg2: tensor<2000xf32>, %arg3: tensor<2000xf32>, %arg4: tensor<2000x2000xf32>) -> (tensor<2000xf32>, tensor<2000xf32>) attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg4, %arg2 : (tensor<2000x2000xf32>, tensor<2000xf32>) -> tensor<2000xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2000xf32>
    %2 = stablehlo.dot %arg3, %arg4 : (tensor<2000xf32>, tensor<2000x2000xf32>) -> tensor<2000xf32>
    %3 = stablehlo.add %arg1, %2 : tensor<2000xf32>
    return %1, %3 : tensor<2000xf32>, tensor<2000xf32>
  }
}

