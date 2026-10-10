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

// Batched NTT: |batch| independent transforms of 2^|lg_domain_size| elements
// each, stored back to back in |inout|. sppark has no fused batched kernels,
// so the transforms are issued round-robin over |nstreams| of the GPU's
// streams (1 through 3), each with its own host->device copy, kernels and
// device->host copy, which lets copies and kernels of different transforms
// overlap. nstreams == 1 runs the transforms one after another.
static RustError batch_ntt(const gpu_t& gpu, fr_t* inout,
                           uint32_t lg_domain_size, size_t batch,
                           size_t nstreams,
                           NTT::InputOutputOrder ntt_order,
                           NTT::Direction ntt_direction,
                           NTT::Type ntt_type,
                           size_t iters, float* elapsed_ms)
{
    if (lg_domain_size == 0 || batch == 0)
        return RustError{cudaSuccess};

    nstreams = std::max<size_t>(1, std::min(nstreams, gpu_t::FLIP_FLOP));

    try {
        gpu.select();

        size_t domain_size = (size_t)1 << lg_domain_size;
        dev_ptr_t<fr_t> d_inout{domain_size * batch};

        if (elapsed_ms == nullptr) {
            for (size_t b = 0; b < batch; b++) {
                auto& stream = gpu[b % nstreams];
                fr_t* d_slice = &d_inout[b * domain_size];
                fr_t* h_slice = &inout[b * domain_size];
                stream.HtoD(d_slice, h_slice, domain_size);
                NTT::Base_dev_ptr(stream, d_slice, lg_domain_size,
                                  ntt_order, ntt_direction, ntt_type);
                stream.DtoH(h_slice, d_slice, domain_size);
            }
            gpu.sync();
        } else {
            // Kernels only: upload once, time |iters| rounds of |batch|
            // transforms between events on the default stream.
            gpu.HtoD(&d_inout[0], inout, domain_size * batch);

            stream_t& zero = gpu;
            cuda_timer_t timer;
            timer.start(zero);
            {
                event_t start{zero};
                for (size_t i = 0; i < nstreams; i++)
                    gpu[i].wait(start);
            }
            for (size_t i = 0; i < iters; i++)
                for (size_t b = 0; b < batch; b++)
                    NTT::Base_dev_ptr(gpu[b % nstreams],
                                      &d_inout[b * domain_size],
                                      lg_domain_size,
                                      ntt_order, ntt_direction, ntt_type);
            for (size_t i = 0; i < nstreams; i++) {
                event_t done{gpu[i]};
                zero.wait(done);
            }
            *elapsed_ms = timer.get_elapsed();

            gpu.DtoH(inout, &d_inout[0], domain_size * batch);
            gpu.sync();
        }
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

SPPARK_FFI
RustError::by_value compute_ntt_batch(size_t device_id,
                                      fr_t* inout, uint32_t lg_domain_size,
                                      size_t batch, size_t nstreams,
                                      NTT::InputOutputOrder ntt_order,
                                      NTT::Direction ntt_direction,
                                      NTT::Type ntt_type)
{
    return batch_ntt(select_gpu(device_id), inout, lg_domain_size,
                     batch, nstreams, ntt_order, ntt_direction, ntt_type,
                     0, nullptr);
}

SPPARK_FFI
RustError::by_value bench_ntt_batch_device(size_t device_id,
                                           fr_t* inout,
                                           uint32_t lg_domain_size,
                                           size_t batch, size_t nstreams,
                                           NTT::InputOutputOrder ntt_order,
                                           NTT::Direction ntt_direction,
                                           NTT::Type ntt_type,
                                           size_t iters, float* elapsed_ms)
{
    *elapsed_ms = 0;
    if (iters == 0)
        return RustError{cudaSuccess};

    return batch_ntt(select_gpu(device_id), inout, lg_domain_size,
                     batch, nstreams, ntt_order, ntt_direction, ntt_type,
                     iters, elapsed_ms);
}
