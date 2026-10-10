// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Host-side instantiation of sppark's multi-threaded Pippenger
// (msm/pippenger.hpp), for benchmarking MSM on a multi-core CPU.

#include <cstddef>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#else
# error "no FEATURE"
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

// Arkworks affine layout, {X, Y, infinity}. Host-side templates expect
// affine_t, {X, Y} with (0, 0) as infinity, so points are converted
// before the MSM.
typedef bucket_t::affine_inf_t ffi_affine_t;

#include <msm/pippenger.hpp>

#include <memory>

static_assert(sizeof(scalar_t::pow_t) == 32, "unexpected scalar size");

// |nthreads| == 0 means "all CPUs in the process' affinity mask",
// |nthreads| == 1 runs single-threaded without spawning a pool.
extern "C"
void mult_pippenger_cpu(point_t* out, const ffi_affine_t points[], size_t npoints,
                        const scalar_t scalars[], size_t nthreads)
{
    // Keep the pool across calls so that thread creation isn't timed.
    // Consequently this function is not reentrant.
    static std::unique_ptr<thread_pool_t> pool;
    static size_t pool_sz = 1;
    if (nthreads != pool_sz) {
        pool.reset(nthreads != 1 ? new thread_pool_t((unsigned int)nthreads)
                                 : nullptr);
        pool_sz = nthreads;
    }

    std::unique_ptr<affine_t[]> converted(new affine_t[npoints]);
    if (pool && npoints >= 1024)
        pool->par_map(npoints, 1024, [&](size_t i) {
            converted[i] = points[i];
        });
    else
        for (size_t i = 0; i < npoints; i++)
            converted[i] = points[i];

    // Arkworks hands over canonical (non-Montgomery) BigInteger256 scalars.
    mult_pippenger<bucket_t>(*out, &converted[0], npoints, scalars, false,
                             pool.get());
}

extern "C"
size_t mult_pippenger_cpu_affine_sz()
{   return sizeof(ffi_affine_t);   }
