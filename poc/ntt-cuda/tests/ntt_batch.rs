// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sppark::{NTTDirection, NTTInputOutputOrder};

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
fn random_vec(len: usize) -> Vec<impl Clone + PartialEq + core::fmt::Debug> {
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
#[test]
fn batch_matches_individual_ntts() {
    for lg_domain_size in [1, 5, 10, 16] {
        let domain_size = 1usize << lg_domain_size;
        for batch in [1, 5, 16] {
            let v = random_vec(domain_size * batch);

            let mut expected = v.clone();
            for chunk in expected.chunks_mut(domain_size) {
                ntt_cuda::NTT(DEFAULT_GPU, chunk, NTTInputOutputOrder::NN);
            }

            for nstreams in 1..=3 {
                let mut vtest = v.clone();
                ntt_cuda::NTT_batch(
                    DEFAULT_GPU,
                    &mut vtest,
                    batch,
                    nstreams,
                    NTTInputOutputOrder::NN,
                    NTTDirection::Forward,
                );
                assert!(vtest == expected);

                ntt_cuda::NTT_batch(
                    DEFAULT_GPU,
                    &mut vtest,
                    batch,
                    nstreams,
                    NTTInputOutputOrder::NN,
                    NTTDirection::Inverse,
                );
                assert!(vtest == v);

                let mut vtest = v.clone();
                ntt_cuda::bench_NTT_batch_device(
                    DEFAULT_GPU,
                    &mut vtest,
                    batch,
                    nstreams,
                    NTTInputOutputOrder::NN,
                    NTTDirection::Forward,
                    1,
                );
                assert!(vtest == expected);
            }
        }
    }
}
