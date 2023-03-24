func.func @main(%arg0: tensor<f32>, %arg1: tensor<1400x1200xf32>) -> tensor<1200x1200xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<1400x1200xf32>, tensor<f32>) -> tensor<1200xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %9 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %9 : tensor<f32>
  }
  %2 = chlo.broadcast_divide %1, %arg0 : (tensor<1200xf32>, tensor<f32>) -> tensor<1200xf32>
  %3 = chlo.broadcast_subtract %arg1, %2 : (tensor<1400x1200xf32>, tensor<1200xf32>) -> tensor<1400x1200xf32>
  %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<1400x1200xf32>) -> tensor<1200x1400xf32>
  %5 = stablehlo.dot %4, %3 : (tensor<1200x1400xf32>, tensor<1400x1200xf32>) -> tensor<1200x1200xf32>
  %6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %7 = stablehlo.subtract %arg0, %6 : tensor<f32>
  %8 = chlo.broadcast_divide %5, %7 : (tensor<1200x1200xf32>, tensor<f32>) -> tensor<1200x1200xf32>
  return %8 : tensor<1200x1200xf32>
}
