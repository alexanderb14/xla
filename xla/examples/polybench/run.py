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
    # print(' '.join(x))
    p = subprocess.run(x, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    return p.stdout.decode('utf-8'), p.stderr.decode('utf-8'), p.returncode


def prepare_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def get_array_dump(s):
    return s.split('==BEGIN DUMP_ARRAYS==')[1].split('==END DUMP_ARRAYS==')[0]


def benchmark_polly(benchmark, validate=False):
    prepare_dir(tmp_dir)
    cmd = ['clang', '-O3', '-march=native',
           '-mllvm', '-polly',
           '-mllvm', '-polly-pattern-matching-based-opts=true',
           '-mllvm', '-polly-parallel',
           '-DPOLYBENCH_USE_SCALAR_LB',
           '-DEXTRALARGE_DATASET',
           '-DPOLYBENCH_TIME',
           '-lgomp',
           '-o', os.path.join(tmp_dir, 'a.out')]
    cmd += benchmark.includes + benchmark.sources
    if validate:
        cmd += ['-DPOLYBENCH_DUMP_ARRAYS']

    assert run_program(cmd)[2] == 0

    cmd = [os.path.join(tmp_dir, 'a.out')]
    out, err, ret = run_program(cmd)

    return out, err


def benchmark_xla(benchmark, validate=False):
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

    out, err, ret = run_program(cmd)

    return out, err


Benchmark = collections.namedtuple(
    'Benchmark', ['name', 'includes', 'sources'])
benchmarks = [
    Benchmark(
        'doitgen', ['-I%s/utilities' % polybench_dir], [
            '%s/linear-algebra/kernels/doitgen/doitgen.c' % polybench_dir,
            '%s/utilities/polybench.c' % polybench_dir])
]

def main():
    for benchmark in tqdm.tqdm(benchmarks):
        print('Benchmarking %s' % benchmark.name)
        prepare_dir(res_dir)

        # Time.
        out_time_polly = float(benchmark_polly(benchmark, validate=False)[0])
        out_time_xla = float(benchmark_xla(benchmark, validate=False)[0])
        print(out_time_polly)
        print(out_time_xla)

        # Validate.
        out_validate_polly = benchmark_polly(benchmark, validate=True)[1]
        out_validate_xla = benchmark_xla(benchmark, validate=True)[1]
        assert get_array_dump(out_validate_polly) == get_array_dump(out_validate_xla)


if __name__ == '__main__':
    main()
