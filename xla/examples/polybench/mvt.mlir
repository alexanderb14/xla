module {
  func.func @main(%arg0: tensor<4000xf64>, %arg1: tensor<4000xf64>, %arg2: tensor<4000xf64>, %arg3: tensor<4000xf64>, %arg4: tensor<4000x4000xf64>) -> (tensor<4000xf64>, tensor<4000xf64>) attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg4, %arg2 : (tensor<4000x4000xf64>, tensor<4000xf64>) -> tensor<4000xf64>
    %1 = stablehlo.add %arg0, %0 : tensor<4000xf64>
    %2 = stablehlo.dot %arg3, %arg4 : (tensor<4000xf64>, tensor<4000x4000xf64>) -> tensor<4000xf64>
    %3 = stablehlo.add %arg1, %2 : tensor<4000xf64>
    return %1, %3 : tensor<4000xf64>, tensor<4000xf64>
  }
}

