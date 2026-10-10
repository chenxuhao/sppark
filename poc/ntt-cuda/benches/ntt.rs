// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use sppark::{NTTDirection, NTTInputOutputOrder, NTTType};

use std::str::FromStr;

const DEFAULT_GPU: usize = 0;

#[cfg(feature = "gl64")]
fn random_vec(len: usize) -> Vec<u64> {
    (0..len)
        .map(|_| rand::random::<u64>() % 0xffffffff00000001)
        .collect()
}

#[cfg(feature = "bb31")]
fn random_vec(len: usize) -> Vec<u32> {
    (0..len)
        .map(|_| rand::random::<u32>() % 0x78000001)
        .collect()
}

#[cfg(not(any(feature = "gl64", feature = "bb31")))]
fn random_vec(len: usize) -> Vec<impl Clone> {
    #[cfg(feature = "bls12_377")]
    use ark_bls12_377::Fr;
    #[cfg(feature = "bls12_381")]
    use ark_bls12_381::Fr;
    #[cfg(feature = "bn254")]
    use ark_bn254::Fr;
    #[cfg(feature = "pallas")]
    use ark_pallas::Fr;
    #[cfg(feature = "vesta")]
    use ark_vesta::Fr;
    use ark_std::UniformRand;

    let rng = &mut ark_std::test_rng();
    (0..len).map(|_| Fr::rand(rng)).collect::<Vec<Fr>>()
}

// BENCH_NPOW: comma-separated log2 domain sizes, default "16,20,24".
//
// For each size two numbers are reported for a forward NTT with natural
// input and output order (NN):
//   e2e/2**k    - ntt_cuda::NTT() as an application would call it: device
//                 allocation, host->device copy, kernels, device->host copy;
//   kernel/2**k - GPU time of the kernels alone on device-resident data,
//                 measured with CUDA events.
fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow =
        std::env::var("BENCH_NPOW").unwrap_or("16,20,24".to_string());

    let mut group = c.benchmark_group("NTT");
    group.sample_size(10);

    for lg_domain_size in bench_npow.split(',') {
        let lg_domain_size = u32::from_str(lg_domain_size.trim()).unwrap();
        let mut v = random_vec(1usize << lg_domain_size);

        group.bench_function(format!("e2e/2**{}", lg_domain_size), |b| {
            b.iter(|| {
                ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NN)
            })
        });

        group.bench_function(format!("kernel/2**{}", lg_domain_size), |b| {
            b.iter_custom(|iters| {
                ntt_cuda::bench_NTT_device(
                    DEFAULT_GPU,
                    &mut v,
                    NTTInputOutputOrder::NN,
                    NTTDirection::Forward,
                    NTTType::Standard,
                    iters as usize,
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
