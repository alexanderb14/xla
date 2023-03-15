func.func @main(%arg0: tensor<1200x1000xf32>, %arg1: tensor<1200x1000xf32>, %arg2: tensor<1200x1200xf32>, %arg3: tensor<f32>) -> tensor<1200x1200xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1200x1000xf32>) -> tensor<1000x1200xf32>
  %1 = stablehlo.dot %arg1, %0 : (tensor<1200x1000xf32>, tensor<1000x1200xf32>) -> tensor<1200x1200xf32>
  %2 = chlo.broadcast_multiply %1, %arg3 : (tensor<1200x1200xf32>, tensor<f32>) -> tensor<1200x1200xf32>
  %3 = stablehlo.add %2, %arg2 : tensor<1200x1200xf32>
  %4 = stablehlo.iota dim = 0 : tensor<1200x1200xi32>
  %5 = stablehlo.constant dense<0> : tensor<i32>
  %6 = stablehlo.constant dense<0> : tensor<1200x1200xi32>
  %7 = stablehlo.add %4, %6 : tensor<1200x1200xi32>
  %8 = stablehlo.iota dim = 1 : tensor<1200x1200xi32>
  %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<1200x1200xi32>, tensor<1200x1200xi32>) -> tensor<1200x1200xi1>
  %10 = stablehlo.select %9, %3, %arg2 : tensor<1200x1200xi1>, tensor<1200x1200xf32>
  return %10 : tensor<1200x1200xf32>
}

