// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_PALLAS)
# include <ff/pasta.hpp>
#elif defined(FEATURE_VESTA)
# include <ff/pasta.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#elif defined(FEATURE_GOLDILOCKS)
# include <ff/goldilocks.hpp>
#elif defined(FEATURE_BABY_BEAR)
# include <ff/baby_bear.hpp>
#else
# error "no FEATURE"
#endif

#include <ntt/ntt.cuh>

SPPARK_FFI
RustError::by_value compute_ntt(size_t device_id,
                                fr_t* inout, uint32_t lg_domain_size,
                                NTT::InputOutputOrder ntt_order,
                                NTT::Direction ntt_direction,
                                NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    return NTT::Base(gpu, inout, lg_domain_size,
                     ntt_order, ntt_direction, ntt_type);
}

#include <util/cuda_timer_t.cuh>

// Benchmarking aid: upload |inout| once, run |iters| back-to-back NTTs on
// the device-resident data and report the elapsed GPU time, as measured
// with CUDA events, excluding allocation and host<->device transfers.
SPPARK_FFI
RustError::by_value bench_ntt_device(size_t device_id,
                                     fr_t* inout, uint32_t lg_domain_size,
                                     NTT::InputOutputOrder ntt_order,
                                     NTT::Direction ntt_direction,
                                     NTT::Type ntt_type,
                                     size_t iters, float* elapsed_ms)
{
    auto& gpu = select_gpu(device_id);

    *elapsed_ms = 0;
    if (lg_domain_size == 0 || iters == 0)
        return RustError{cudaSuccess};

    try {
        gpu.select();

        size_t domain_size = (size_t)1 << lg_domain_size;
        dev_ptr_t<fr_t> d_inout{domain_size, gpu};
        gpu.HtoD(&d_inout[0], inout, domain_size);

        cuda_timer_t timer;
        timer.start(gpu);
        for (size_t i = 0; i < iters; i++)
            NTT::Base_dev_ptr(gpu, &d_inout[0], lg_domain_size,
                              ntt_order, ntt_direction, ntt_type);
        *elapsed_ms = timer.get_elapsed();

        gpu.DtoH(inout, &d_inout[0], domain_size);
        gpu.sync();
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}
