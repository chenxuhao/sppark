// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "bls12_377")]
use ark_bls12_377::G1Affine;
#[cfg(feature = "bls12_381")]
use ark_bls12_381::G1Affine;
#[cfg(feature = "bn254")]
use ark_bn254::G1Affine;
use ark_ec::msm::VariableBaseMSM;
use ark_ff::BigInteger256;

use std::str::FromStr;

use msm_cuda::*;

// BENCH_NPOW:    log2 of the number of points, default 20.
// BENCH_THREADS: comma-separated thread counts for sppark's host-side
//                Pippenger, 0 meaning all CPUs, default "1,0".
// BENCH_ARKWORKS=1 adds arkworks' VariableBaseMSM as a baseline; its
//                thread count is controlled with RAYON_NUM_THREADS.
fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("20".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();
    let bench_threads =
        std::env::var("BENCH_THREADS").unwrap_or("1,0".to_string());

    let (points, scalars) =
        util::generate_points_scalars::<G1Affine>(1usize << npoints_npow);
    let scalars = unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    };

    let mut group = c.benchmark_group("CPU");
    group.sample_size(10);

    for nthreads in bench_threads.split(',') {
        let nthreads = usize::from_str(nthreads.trim()).unwrap();
        let name = format!("sppark/2**{}/threads={}", npoints_npow, nthreads);
        group.bench_function(name, |b| {
            b.iter(|| {
                let _ = multi_scalar_mult_cpu_arkworks(
                    points.as_slice(),
                    scalars,
                    nthreads,
                );
            })
        });
    }

    if std::env::var("BENCH_ARKWORKS").is_ok_and(|v| v == "1") {
        let name = format!("arkworks/2**{}", npoints_npow);
        group.bench_function(name, |b| {
            b.iter(|| {
                let _ = VariableBaseMSM::multi_scalar_mul(
                    points.as_slice(),
                    scalars,
                );
            })
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
