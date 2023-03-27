module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1000x1200xf32>, %arg3: tensor<1000x1000xf32>, %arg4: tensor<1000x1200xf32>) -> tensor<1000x1200xf32> attributes {irsynth.raised} {
    %0 = chlo.broadcast_multiply %arg4, %arg1 : (tensor<1000x1200xf32>, tensor<f32>) -> tensor<1000x1200xf32>
    %1 = chlo.broadcast_multiply %arg0, %arg2 : (tensor<f32>, tensor<1000x1200xf32>) -> tensor<1000x1200xf32>
    %2 = stablehlo.dot %arg3, %1 : (tensor<1000x1000xf32>, tensor<1000x1200xf32>) -> tensor<1000x1200xf32>
    %3 = chlo.broadcast_add %0, %2 : (tensor<1000x1200xf32>, tensor<1000x1200xf32>) -> tensor<1000x1200xf32>
    return %3 : tensor<1000x1200xf32>
  }
}
