module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1000x1100xf32>, %arg3: tensor<1000x1200xf32>, %arg4: tensor<1200x1100xf32>) -> tensor<1000x1100xf32> attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg3, %arg4 : (tensor<1000x1200xf32>, tensor<1200x1100xf32>) -> tensor<1000x1100xf32>
    %1 = chlo.broadcast_multiply %0, %arg0 : (tensor<1000x1100xf32>, tensor<f32>) -> tensor<1000x1100xf32>
    %2 = chlo.broadcast_multiply %arg2, %arg1 : (tensor<1000x1100xf32>, tensor<f32>) -> tensor<1000x1100xf32>
    %3 = stablehlo.add %2, %1 : tensor<1000x1100xf32>
    return %3 : tensor<1000x1100xf32>
  }
}
