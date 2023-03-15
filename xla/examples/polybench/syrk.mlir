func.func @main(%arg0: tensor<1200x1200xf32> {irsynth.symmetric}, %arg1: tensor<1200x1000xf32>, %arg2: tensor<f32>, %arg1000: tensor<f32>) -> tensor<1200x1200xf32> {
  %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<1200x1000xf32>) -> tensor<1000x1200xf32>
  %1 = stablehlo.dot %arg1, %0 : (tensor<1200x1000xf32>, tensor<1000x1200xf32>) -> tensor<1200x1200xf32>
  %2 = chlo.broadcast_multiply %1, %arg1000 : (tensor<1200x1200xf32>, tensor<f32>) -> tensor<1200x1200xf32>
  %3 = chlo.broadcast_multiply %arg0, %arg2 : (tensor<1200x1200xf32>, tensor<f32>) -> tensor<1200x1200xf32>
  %4 = stablehlo.add %3, %2 : tensor<1200x1200xf32>
  %5 = stablehlo.iota dim = 0 : tensor<1200x1200xi32>
  %6 = stablehlo.constant dense<0> : tensor<i32>
  %7 = stablehlo.constant dense<0> : tensor<1200x1200xi32>
  %8 = stablehlo.add %5, %7 : tensor<1200x1200xi32>
  %9 = stablehlo.iota dim = 1 : tensor<1200x1200xi32>
  %10 = stablehlo.compare  GE, %8, %9,  SIGNED : (tensor<1200x1200xi32>, tensor<1200x1200xi32>) -> tensor<1200x1200xi1>
  %11 = stablehlo.select %10, %4, %arg0 : tensor<1200x1200xi1>, tensor<1200x1200xf32>
  return %11 : tensor<1200x1200xf32>
}
