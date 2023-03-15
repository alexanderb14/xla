func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1200x1200xf32>, %arg3: tensor<1200x1000xf32>, %arg4: tensor<1200x1000xf32>) -> tensor<1200x1200xf32> {
  %0 = chlo.broadcast_multiply %arg2, %arg1 : (tensor<1200x1200xf32>, tensor<f32>) -> tensor<1200x1200xf32>
  %1 = chlo.broadcast_multiply %arg4, %arg0 : (tensor<1200x1000xf32>, tensor<f32>) -> tensor<1200x1000xf32>
  %2 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<1200x1000xf32>) -> tensor<1000x1200xf32>
  %3 = stablehlo.dot %1, %2 : (tensor<1200x1000xf32>, tensor<1000x1200xf32>) -> tensor<1200x1200xf32>
  %4 = chlo.broadcast_multiply %arg3, %arg0 : (tensor<1200x1000xf32>, tensor<f32>) -> tensor<1200x1000xf32>
  %5 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<1200x1000xf32>) -> tensor<1000x1200xf32>
  %6 = stablehlo.dot %4, %5 : (tensor<1200x1000xf32>, tensor<1000x1200xf32>) -> tensor<1200x1200xf32>
  %7 = stablehlo.add %0, %3 : tensor<1200x1200xf32>
  %8 = stablehlo.add %7, %6 : tensor<1200x1200xf32>
  %9 = stablehlo.iota dim = 0 : tensor<1200x1200xi32>
  %10 = stablehlo.constant dense<0> : tensor<i32>
  %11 = stablehlo.constant dense<0> : tensor<1200x1200xi32>
  %12 = stablehlo.add %9, %11 : tensor<1200x1200xi32>
  %13 = stablehlo.iota dim = 1 : tensor<1200x1200xi32>
  %14 = stablehlo.compare  GE, %12, %13,  SIGNED : (tensor<1200x1200xi32>, tensor<1200x1200xi32>) -> tensor<1200x1200xi1>
  %15 = stablehlo.select %14, %8, %arg2 : tensor<1200x1200xi1>, tensor<1200x1200xf32>
  return %15 : tensor<1200x1200xf32>
}
