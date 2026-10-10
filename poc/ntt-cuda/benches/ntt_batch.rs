// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use sppark::{NTTDirection, NTTInputOutputOrder};

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
// BENCH_NPOW:    comma-separated log2 sizes of each transform,
//                default "16,20,22".
// BENCH_BATCH:   number of transforms per batch, default 16.
// BENCH_STREAMS: comma-separated GPU stream counts (1 through 3) to spread
//                the batch over, default "1,3"; 1 means one after another.
//
// For each combination two numbers are reported for a whole batch of
// forward NN NTTs:
//   e2e    - ntt_cuda::NTT_batch(): allocation, host->device copies,
//            kernels, device->host copies;
//   kernel - GPU time of the kernels alone on device-resident data,
//            measured with CUDA events.
fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow =
        std::env::var("BENCH_NPOW").unwrap_or("16,20,22".to_string());
    let batch = std::env::var("BENCH_BATCH").unwrap_or("16".to_string());
    let batch = usize::from_str(&batch).unwrap();
    let bench_streams =
        std::env::var("BENCH_STREAMS").unwrap_or("1,3".to_string());

    let mut group = c.benchmark_group("NTT-batch");
    group.sample_size(10);

    for lg_domain_size in bench_npow.split(',') {
        let lg_domain_size = u32::from_str(lg_domain_size.trim()).unwrap();
        let mut v = random_vec(batch << lg_domain_size);

        for nstreams in bench_streams.split(',') {
            let nstreams = usize::from_str(nstreams.trim()).unwrap();
            let name = format!(
                "{}x2**{}/streams={}",
                batch, lg_domain_size, nstreams
            );

            group.bench_function(format!("e2e/{}", name), |b| {
                b.iter(|| {
                    ntt_cuda::NTT_batch(
                        DEFAULT_GPU,
                        &mut v,
                        batch,
                        nstreams,
                        NTTInputOutputOrder::NN,
                        NTTDirection::Forward,
                    )
                })
            });

            group.bench_function(format!("kernel/{}", name), |b| {
                b.iter_custom(|iters| {
                    ntt_cuda::bench_NTT_batch_device(
                        DEFAULT_GPU,
                        &mut v,
                        batch,
                        nstreams,
                        NTTInputOutputOrder::NN,
                        NTTDirection::Forward,
                        iters as usize,
                    )
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
