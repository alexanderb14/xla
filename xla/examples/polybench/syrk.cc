#include "xla_compiler.h"
#include "polybench.h"
#include "syrk.h"

#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

using namespace xla;

/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syrk(int n, int m,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;

//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_M; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  cmd_option option = parseOption(argc, argv);

  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Prepare computation. */
  // - Build executable
  auto client = buildJITClient(option);
  auto executable = buildExecutable(
      client, "/devel/git_3rd/xla/xla/examples/polybench/syrk.mlir");

  // - Create inputs.
  auto alpha_b = buildBufferFromScalar(client, alpha);

  auto beta_b = buildBufferFromScalar(client, beta);
  //
  auto C_a = xla::Array2D<float>(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C_a(i, j) = (*C)[i][j];
    }
  }
  auto C_b = buildBuffer2D(client, C_a);

  auto A_a = xla::Array2D<float>(n, m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      A_a(i, j) = (*A)[i][j];
    }
  }
  auto A_b = buildBuffer2D(client, A_a);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  ::xla::ExecuteOptions options;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result =
      executable->Execute({{C_b.get(), A_b.get(), alpha_b.get(), beta_b.get()}},
                          options).value();

  auto buffer = result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  /* Stop and print timer. */
  polybench_timer_stop();
  if (option == option_time || option == option_time_sequential)
    polybench_timer_print();

  /* Store the result data. */
  std::shared_ptr<Literal> result_literal = 
                          result[0][0]->ToLiteralSync().value();
  auto result_a = result_literal->data<float>();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      (*C)[i][j] = result_a[i * n + j];
    }
  }

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (option == option_validate)
    print_array(n, POLYBENCH_ARRAY(C));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}
