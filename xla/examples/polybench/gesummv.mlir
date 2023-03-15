module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1300x1300xf32>, %arg3: tensor<1300x1300xf32>, %arg4: tensor<1300xf32>, %arg5: tensor<1300xf32>, %arg6: tensor<1300xf32>) -> tensor<1300xf32> attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg2, %arg5 : (tensor<1300x1300xf32>, tensor<1300xf32>) -> tensor<1300xf32>
    %1 = chlo.broadcast_multiply %0, %arg0 : (tensor<1300xf32>, tensor<f32>) -> tensor<1300xf32>
    %2 = stablehlo.dot %arg3, %arg5 : (tensor<1300x1300xf32>, tensor<1300xf32>) -> tensor<1300xf32>
    %3 = chlo.broadcast_multiply %2, %arg1 : (tensor<1300xf32>, tensor<f32>) -> tensor<1300xf32>
    %4 = stablehlo.add %1, %3 : tensor<1300xf32>
    return %4 : tensor<1300xf32>
  }
}
