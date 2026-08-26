/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver smoke test for persisting-L2 access-policy windows.

use cuda_core::{ContextLimit, CudaContext, DeviceBuffer};

#[test]
fn persisting_l2_window_applies_and_clears() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let limits = ctx
        .persisting_l2_cache_limits()
        .expect("failed to query persisting-L2 limits");
    if limits.max_persisting_l2_cache_bytes == 0 || limits.max_access_policy_window_bytes == 0 {
        eprintln!("skipping: device does not expose persisting-L2 access-policy windows");
        return;
    }

    let original = ctx
        .limit(ContextLimit::PersistingL2CacheSize)
        .expect("failed to read the persisting-L2 reservation");
    let reservation = limits.max_persisting_l2_cache_bytes.min(1 << 20);
    ctx.set_limit(ContextLimit::PersistingL2CacheSize, reservation)
        .expect("failed to reserve persisting-L2 capacity");

    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let buffer = DeviceBuffer::<u8>::zeroed(&stream, 4096)
        .expect("failed to allocate the access-policy buffer");
    let applied = stream
        .set_persisting_access_policy(&buffer, usize::MAX, 0.5)
        .expect("failed to apply a persisting-L2 access-policy window");
    assert_eq!(
        applied,
        buffer
            .num_bytes()
            .min(limits.max_access_policy_window_bytes)
    );
    stream
        .clear_access_policy_window()
        .expect("failed to clear the persisting-L2 access-policy window");
    stream.synchronize().expect("failed to synchronize stream");

    ctx.set_limit(ContextLimit::PersistingL2CacheSize, original)
        .expect("failed to restore the persisting-L2 reservation");
}
