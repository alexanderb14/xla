#include "xla_compiler.h"
#include "polybench.h"
#include "doitgen.h"

#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

using namespace xla;

/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++)  {
      for (p = 0; p < _PB_NP; p++)  {
	sum[p] = SCALAR_VAL(0.0);
	for (s = 0; s < _PB_NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < _PB_NP; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop

}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  /* Prepare computation. */
  // - Build executable
  auto client = buildJITClient();
  auto executable = buildExecutable(
      client, "/devel/git_3rd/xla/xla/examples/axpy/doitgen.mlir");

  // - Create inputs.
  auto x_a = xla::Array3D<double>(NR, NQ, NP);
  for (int i = 0; i < x_a.dim(0); ++i)
    for (int j = 0; j < x_a.dim(1); ++j)
      for (int k = 0; k < x_a.dim(2); ++k)
	      x_a(i, j, k) = (*A)[i][j][k];
  auto x = buildBuffer3D(client, x_a);

  auto y_a = xla::Array2D<double>(NP, NP);
  for (int i = 0; i < y_a.dim(0); ++i)
    for (int j = 0; j < y_a.dim(1); ++j)
      y_a(i, j) = (*C4)[i][j];
  auto y = buildBuffer2D(client, y_a);

  /* Start timer. */
  polybench_start_instruments;

  ::xla::ExecuteOptions options;
  options.execution_mode = ::xla::ExecuteOptions::ExecutionMode::kSynchronous;

  /* Run kernel. */
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> axpy_result =
      executable->Execute({{x.get(), y.get()}}, options).value();

  auto buffer = axpy_result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Store the result data. */
  std::shared_ptr<Literal> axpy_result_literal = 
                          axpy_result[0][0]->ToLiteralSync().value();
  auto axpy_result_a = axpy_result_literal->data<double>();
  for (int i = 0; i < x_a.dim(0); ++i)
    for (int j = 0; j < x_a.dim(1); ++j)
      for (int k = 0; k < x_a.dim(2); ++k)
        (*A)[i][j][k] = axpy_result_a[i*x_a.dim(1)*x_a.dim(2) + j*x_a.dim(2) + k];

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}

