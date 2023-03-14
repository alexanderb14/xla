#include "xla_compiler.h"
#include "polybench.h"
#include "bicg.h"

#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

using namespace xla;

/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_M; i++)
    s[i] = 0;
  for (i = 0; i < _PB_N; i++)
    {
      q[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_M; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Prepare computation. */
  // - Build executable
  auto client = buildJITClient(option);
  auto executable = buildExecutable(
      client, "/devel/git_3rd/xla/xla/examples/polybench/bicg.mlir");

  // - Create inputs.
  auto A_a = xla::Array2D<double>(n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A_a(i, j) = (*A)[i][j];
    }
  }
  auto A_b = buildBuffer1D(client, A_a);

  auto s_a = xla::Array<double>({m});
  for (int i = 0; i < m; ++i) {
    s_a(i) = (*s)[i];
  }
  auto s_b = buildBuffer1D(client, s_a);

  auto q_a = xla::Array<double>({n});
  for (int i = 0; i < n; ++i) {
    q_a(i) = (*q)[i];
  }
  auto q_b = buildBuffer1D(client, q_a);

  auto p_a = xla::Array<double>({m});
  for (int i = 0; i < m; ++i) {
    p_a(i) = (*p)[i];
  }
  auto p_b = buildBuffer1D(client, p_a);

  auto r_a = xla::Array<double>({n});
  for (int i = 0; i < n; ++i) {
    r_a(i) = (*r)[i];
  }
  auto r_b = buildBuffer1D(client, r_a);

  /* Run kernel. */
  ::xla::ExecuteOptions options;
  executable->Execute({{A_b.get(), s_b.get(), q_b.get(), p_b.get(), r_b.get()}}, options).value();
  executable->Execute({{A_b.get(), s_b.get(), q_b.get(), p_b.get(), r_b.get()}}, options).value();

  /* Start timer. */
  polybench_start_instruments;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result =
      executable->Execute({{A_b.get(), s_b.get(), q_b.get(), p_b.get(), r_b.get()}}, options).value();

  auto buffer = result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  /* Stop and print timer. */
  polybench_timer_stop();
  if (option == option_time || option == option_time_sequential)
    polybench_timer_print();

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (option == option_validate)
    print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}
