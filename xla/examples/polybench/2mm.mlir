func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<800x900xf32>, %arg3: tensor<800x1100xf32>, %arg4: tensor<1100x900xf32>, %arg5: tensor<900x1200xf32>, %arg6: tensor<800x1200xf32>) -> tensor<800x1200xf32> {
  %0 = chlo.broadcast_multiply %arg4, %arg0 : (tensor<1100x900xf32>, tensor<f32>) -> tensor<1100x900xf32>
  %1 = stablehlo.dot %arg3, %0 : (tensor<800x1100xf32>, tensor<1100x900xf32>) -> tensor<800x900xf32>
  %2 = stablehlo.dot %1, %arg5 : (tensor<800x900xf32>, tensor<900x1200xf32>) -> tensor<800x1200xf32>
  %3 = chlo.broadcast_multiply %arg6, %arg1 : (tensor<800x1200xf32>, tensor<f32>) -> tensor<800x1200xf32>
  %4 = stablehlo.add %3, %2 : tensor<800x1200xf32>
  return %4 : tensor<800x1200xf32>
}
