import argparse
import collections
import json
import multiprocessing
import os
import shutil
import subprocess
import time

import tqdm
import pandas as pd

polybench_dir = '/devel/git/irSynth-eval/benchmarks/polybench-c-4.2.1-beta'
res_dir = '/tmp/perf_results'
tmp_dir = '/tmp/perf_results_tmp'

def run_program(x):
    print(' '.join(x))
    p = subprocess.run(x, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    return p.stdout.decode('utf-8'), p.stderr.decode('utf-8'), p.returncode


def prepare_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def get_array_dump(s):
    return s.split('==BEGIN DUMP_ARRAYS==')[1].split('==END DUMP_ARRAYS==')[0]


def benchmark_O3(benchmark, validate=False):
    prepare_dir(tmp_dir)
    cmd = ['clang', '-O3', '-march=native',
           '-DPOLYBENCH_USE_SCALAR_LB',
           '-DEXTRALARGE_DATASET',
           '-DPOLYBENCH_TIME',
           '-o', os.path.join(tmp_dir, 'a.out')]
    cmd += benchmark.includes + benchmark.sources
    if validate:
        cmd += ['-DPOLYBENCH_DUMP_ARRAYS']

    assert run_program(cmd)[2] == 0

    cmd = [os.path.join(tmp_dir, 'a.out')]
    out, err, ret = run_program(cmd)

    return out, err


def benchmark_polly(benchmark, parallel, validate=False):
    prepare_dir(tmp_dir)
    cmd = ['clang', '-O3', '-march=native',
           '-DPOLYBENCH_USE_SCALAR_LB',
           '-DEXTRALARGE_DATASET',
           '-DPOLYBENCH_TIME',
           '-mllvm', '-polly',
           '-mllvm', '-polly-pattern-matching-based-opts=true',
           '-o', os.path.join(tmp_dir, 'a.out')]
    if parallel:
        cmd += ['-mllvm', '-polly-parallel', '-lgomp']
    cmd += benchmark.includes + benchmark.sources
    if validate:
        cmd += ['-DPOLYBENCH_DUMP_ARRAYS']

    assert run_program(cmd)[2] == 0

    cmd = [os.path.join(tmp_dir, 'a.out')]
    out, err, ret = run_program(cmd)

    return out, err


def benchmark_xla(benchmark, parallel, validate=False):
    cmd = ['bazel', 'build',
            'xla/examples/polybench:%s' % benchmark.name,
            '--nocheck_visibility',
            '--test_output=all',
            '--config=avx_linux',
            '--define=tensorflow_mkldnn_contraction_kernel=1',
            '--verbose_failures']
    assert run_program(cmd)[2] == 0

    cmd = ['./bazel-bin/xla/examples/polybench/%s' % benchmark.name]
    if validate:
        cmd += ['--validate']
    if not parallel:
        cmd += ['--time-sequential']

    out, err, ret = run_program(cmd)

    return out, err


Benchmark = collections.namedtuple(
    'Benchmark', ['name', 'includes', 'sources'])
benchmarks = [
    Benchmark(
        'bicg', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/bicg/bicg.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        'gemver', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/blas/gemver/gemver.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        'gesummv', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/blas/gesummv/gesummv.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        'gemm', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/blas/gemm/gemm.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        'atax', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/atax/atax.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        'mvt', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/mvt/mvt.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        '2mm', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/2mm/2mm.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        '3mm', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/3mm/3mm.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
    Benchmark(
        'doitgen', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/doitgen/doitgen.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir]),
]

def main():
    prepare_dir(res_dir)

    df = pd.DataFrame(columns=['benchmark', 'compiler', 'time'])

    for benchmark in tqdm.tqdm(benchmarks):
        print('Benchmarking %s' % benchmark.name)

        # Time.
        time_O3 = float(benchmark_O3(benchmark, validate=False)[0])
        print(time_O3)
        time_polly = float(benchmark_polly(benchmark, parallel=False, validate=False)[0])
        print(time_polly)
        time_polly_parallel = float(benchmark_polly(benchmark, parallel=True, validate=False)[0])
        print(time_polly_parallel)
        time_xla = float(benchmark_xla(benchmark, parallel=False, validate=False)[0])
        print(time_xla)
        time_xla_parallel = float(benchmark_xla(benchmark, parallel=True, validate=False)[0])
        print(time_xla_parallel)

        # # Validate.
        # out_validate_polly = get_array_dump(benchmark_polly(benchmark, validate=True)[1])
        # out_validate_xla = get_array_dump(benchmark_xla(benchmark, validate=True)[1])
        # if out_validate_polly == out_validate_xla:
        #     print('VALIDATION SUCCEEDED')
        # else:
        #     print('VALIDATION FAILED')

        # # Write validation outputs to files.
        # with open(os.path.join(res_dir, '%s_polly.txt' % benchmark.name), 'w') as f:
        #     f.write(out_validate_polly)
        # with open(os.path.join(res_dir, '%s_xla.txt' % benchmark.name), 'w') as f:
        #     f.write(out_validate_xla)

        # Write results to dataframe.
        df = pd.concat([df, pd.DataFrame({
            'benchmark': [benchmark.name],
            'compiler': ['llvm-O3'],
            'time': [time_O3]})])
        df = pd.concat([df, pd.DataFrame({
            'benchmark': [benchmark.name],
            'compiler': ['polly'],
            'time': [time_polly]})])
        df = pd.concat([df, pd.DataFrame({
            'benchmark': [benchmark.name],
            'compiler': ['polly-parallel'],
            'time': [time_polly_parallel]})])
        df = pd.concat([df, pd.DataFrame({
            'benchmark': [benchmark.name],
            'compiler': ['xla'],
            'time': [time_xla]})])
        df = pd.concat([df, pd.DataFrame({
            'benchmark': [benchmark.name],
            'compiler': ['xla-parallel'],
            'time': [time_xla_parallel]})])

    df.to_csv(os.path.join(res_dir, 'results.csv'), index=False)

if __name__ == '__main__':
    main()
