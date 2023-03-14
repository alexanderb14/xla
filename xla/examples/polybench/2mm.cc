#include "xla_compiler.h"
#include "polybench.h"
#include "2mm.h"

#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

using namespace xla;

/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	tmp[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < _PB_NJ; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }
#pragma endscop

}


int main(int argc, char** argv)
{
  cmd_option option = parseOption(argc, argv);

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Prepare computation. */
  // - Build executable
  auto client = buildJITClient(option);
  auto executable = buildExecutable(
      client, "/devel/git_3rd/xla/xla/examples/polybench/2mm.mlir");

  // - Create inputs.
  auto alpha_b = buildBufferFromScalar(client, alpha);

  auto beta_b = buildBufferFromScalar(client, beta);

  auto tmp_a = xla::Array2D<double>(ni, nj);
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      tmp_a(i, j) = (*tmp)[i][j];
    }
  }
  auto tmp_b = buildBuffer2D(client, tmp_a);

  auto A_a = xla::Array2D<double>(ni, nk);
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nk; j++)
      A_a(i, j) = (*A)[i][j];
  auto A_b = buildBuffer2D(client, A_a);

  auto B_a = xla::Array2D<double>(nk, nj);
  for (int i = 0; i < nk; i++)
    for (int j = 0; j < nj; j++)
      B_a(i, j) = (*B)[i][j];
  auto B_b = buildBuffer2D(client, B_a);

  auto C_a = xla::Array2D<double>(nj, nl);
  for (int i = 0; i < nj; i++)
    for (int j = 0; j < nl; j++)
      C_a(i, j) = (*C)[i][j];
  auto C_b = buildBuffer2D(client, C_a);

  auto D_a = xla::Array2D<double>(ni, nl);
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++)
      D_a(i, j) = (*D)[i][j];
  auto D_b = buildBuffer2D(client, D_a);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  ::xla::ExecuteOptions options;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result =
      executable->Execute({{alpha_b.get(), beta_b.get(),
                            tmp_b.get(), A_b.get(), B_b.get(), C_b.get(), D_b.get()}}, options).value();

  auto buffer = result[0][0].get();
  auto status = buffer->BlockHostUntilReady();

  /* Stop and print timer. */
  polybench_timer_stop();
  if (option == option_time || option == option_time_sequential)
    polybench_timer_print();

  /* Store the result data. */
  std::shared_ptr<Literal> result_literal = 
                          result[0][0]->ToLiteralSync().value();
  auto result_a = result_literal->data<double>();
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++)
      (*D)[i][j] = result_a[i * nl + j];

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (option == option_validate)
    print_array(ni, nl,  POLYBENCH_ARRAY(D));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}
