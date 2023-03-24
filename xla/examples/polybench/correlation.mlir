func.func @main(%arg0: tensor<f32>, %arg1: tensor<1400x1200xf32>) -> tensor<1200x1200xf32> attributes {irsynth.raised} {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<1200x1200xf32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<1200xf32>
  %2 = stablehlo.constant dense<1.000000e-01> : tensor<1200xf32>
  %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %4 = stablehlo.reduce(%arg1 init: %3) across dimensions = [0] : (tensor<1400x1200xf32>, tensor<f32>) -> tensor<1200xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %26 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %26 : tensor<f32>
  }
  %5 = chlo.broadcast_divide %4, %arg0 : (tensor<1200xf32>, tensor<f32>) -> tensor<1200xf32>
  %6 = chlo.broadcast_subtract %arg1, %5 : (tensor<1400x1200xf32>, tensor<1200xf32>) -> tensor<1400x1200xf32>
  %7 = stablehlo.multiply %6, %6 : tensor<1400x1200xf32>
  %8 = stablehlo.reduce(%7 init: %3) across dimensions = [0] : (tensor<1400x1200xf32>, tensor<f32>) -> tensor<1200xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %26 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %26 : tensor<f32>
  }
  %9 = chlo.broadcast_divide %8, %arg0 : (tensor<1200xf32>, tensor<f32>) -> tensor<1200xf32>
  %10 = stablehlo.sqrt %9 : tensor<1200xf32>
  %11 = stablehlo.compare  LE, %10, %2,  FLOAT : (tensor<1200xf32>, tensor<1200xf32>) -> tensor<1200xi1>
  %12 = stablehlo.select %11, %1, %10 : tensor<1200xi1>, tensor<1200xf32>
  %13 = chlo.broadcast_subtract %arg1, %5 : (tensor<1400x1200xf32>, tensor<1200xf32>) -> tensor<1400x1200xf32>
  %14 = stablehlo.sqrt %arg0 : tensor<f32>
  %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<1200xf32>
  %16 = stablehlo.multiply %15, %12 : tensor<1200xf32>
  %17 = chlo.broadcast_divide %13, %16 : (tensor<1400x1200xf32>, tensor<1200xf32>) -> tensor<1400x1200xf32>
  %18 = stablehlo.transpose %17, dims = [1, 0] : (tensor<1400x1200xf32>) -> tensor<1200x1400xf32>
  %19 = stablehlo.dot %18, %17, precision = [DEFAULT, DEFAULT] : (tensor<1200x1400xf32>, tensor<1400x1200xf32>) -> tensor<1200x1200xf32>
  %20 = stablehlo.iota dim = 0 : tensor<1200xi32>
  %21 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<1200xi32>) -> tensor<1200x1200xi32>
  %22 = stablehlo.iota dim = 0 : tensor<1200xi32>
  %23 = stablehlo.broadcast_in_dim %22, dims = [1] : (tensor<1200xi32>) -> tensor<1200x1200xi32>
  %24 = stablehlo.compare  EQ, %21, %23,  SIGNED : (tensor<1200x1200xi32>, tensor<1200x1200xi32>) -> tensor<1200x1200xi1>
  %25 = stablehlo.select %24, %0, %19 : tensor<1200x1200xi1>, tensor<1200x1200xf32>
  return %25 : tensor<1200x1200xf32>
}
