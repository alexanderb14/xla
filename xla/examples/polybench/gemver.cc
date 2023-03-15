#include "xla_compiler.h"
#include "polybench.h"
#include "gemver.h"

#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

using namespace xla;

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}


int main(int argc, char** argv)
{
  cmd_option option = parseOption(argc, argv);

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Prepare computation. */
  // - Build executable
  auto client = buildJITClient(option);
  auto executable = buildExecutable(
      client, "/devel/git_3rd/xla/xla/examples/polybench/gemver.mlir");

  // - Create inputs.
  auto alpha_b = buildBufferFromScalar(client, alpha);

  auto beta_b = buildBufferFromScalar(client, beta);

  auto A_a = xla::Array2D<float>(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A_a(i, j) = (*A)[i][j];
  auto A_b = buildBuffer2D(client, A_a);

  auto u1_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    u1_a(i) = (*u1)[i];
  auto u1_b = buildBuffer1D(client, u1_a);

  auto v1_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    v1_a(i) = (*v1)[i];
  auto v1_b = buildBuffer1D(client, v1_a);

  auto u2_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    u2_a(i) = (*u2)[i];
  auto u2_b = buildBuffer1D(client, u2_a);

  auto v2_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    v2_a(i) = (*v2)[i];
  auto v2_b = buildBuffer1D(client, v2_a);

  auto w_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    w_a(i) = (*w)[i];
  auto w_b = buildBuffer1D(client, w_a);

  auto x_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    x_a(i) = (*x)[i];
  auto x_b = buildBuffer1D(client, x_a);

  auto y_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    y_a(i) = (*y)[i];
  auto y_b = buildBuffer1D(client, y_a);

  auto z_a = xla::Array<float>({n});
  for (int i = 0; i < n; i++)
    z_a(i) = (*z)[i];
  auto z_b = buildBuffer1D(client, z_a);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  ::xla::ExecuteOptions options;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result =
      executable->Execute({{alpha_b.get(), beta_b.get(),
                            A_b.get(), u1_b.get(), v1_b.get(), u2_b.get(),
                            v2_b.get(), w_b.get(), x_b.get(), y_b.get(),
                            z_b.get()}},
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
  for (int i = 0; i < n; i++)
    (*w)[i] = result_a[i];

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (option == option_validate)
    print_array(n, POLYBENCH_ARRAY(w));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}
