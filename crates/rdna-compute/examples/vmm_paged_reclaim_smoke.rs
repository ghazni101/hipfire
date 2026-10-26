// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Physical reclaim for the shared VMM page pool.
//!
//! One granule group is the budget. The pool first resides on the high
//! granule, then one sync moves residency to the low granule. Mapping the
//! new granule before releasing the old one exceeds the budget and must
//! not: free granules are unmapped first, matching a block allocator that
//! returns memory before the next reservation. Releasing every page then
//! returns the physical bytes.
//!
//! Run:
//!   cargo run -p rdna-compute --example vmm_paged_reclaim_smoke --features lab

use rdna_compute::page_pool::PageState;
use rdna_compute::vmm_page_pool::{VmmGranuleGeometry, VmmPageMapper};
use rdna_compute::Gpu;

fn dev() -> i32 {
    std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init_with_device(dev())?;
    // 128 * 16 KiB = 2 MiB, so the default granule holds exactly one page.
    // Two pages are two granule groups. The budget is one group.
    const PAGE_TOKENS: usize = 128;
    const PER_POS: usize = 16 * 1024;
    let granularity = gpu.vmm_recommended_granularity()?;
    let geom = VmmGranuleGeometry::plan(granularity, PAGE_TOKENS, PER_POS, 0)?;
    assert_eq!(
        geom.pages_per_granule, 1,
        "expected one page per granule, got {} (page {} B, granule {} B)",
        geom.pages_per_granule, geom.page_bytes, geom.granule_bytes
    );
    let group = geom.group_bytes(1)?;
    let (mut mapper, mut arenas) =
        VmmPageMapper::new(&mut gpu, 1, 2, PAGE_TOKENS, PER_POS, 0, group)?;
    assert_eq!(mapper.page_layout().n_granules, 2);
    assert_eq!(mapper.mapped_bytes(), 0, "load is VA-only");

    let high = vec![PageState::Free, PageState::Private];
    mapper.sync(&mut gpu, &high)?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(
        mapper.mapped_bytes(),
        group,
        "only the high granule is resident"
    );
    mapper.check_table(&[1], 1)?;
    assert!(
        mapper.check_table(&[0], 1).is_err(),
        "the low granule is not mapped yet"
    );

    // Same budget, opposite residency. The low granule maps only because
    // the high one is released first.
    let low = vec![PageState::Private, PageState::Free];
    mapper.sync(&mut gpu, &low)?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(
        mapper.mapped_bytes(),
        group,
        "residency moved, it did not double"
    );
    mapper.check_table(&[0], 1)?;
    assert!(
        mapper.check_table(&[1], 1).is_err(),
        "the high granule must be unmapped once its page is Free"
    );

    let offset = mapper.page_offset(0).expect("page 0");
    let pattern = vec![0x5Au8; 4096];
    gpu.hip
        .memcpy_htod_offset(&arenas.k[0].buf, offset, &pattern)?;
    let mut got = vec![0u8; 4096];
    gpu.hip.memcpy_dtoh_at(&mut got, &arenas.k[0].buf, offset)?;
    assert_eq!(
        got, pattern,
        "page 0 is not readable after the granule move"
    );

    let none = vec![PageState::Free, PageState::Free];
    mapper.sync(&mut gpu, &none)?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(
        mapper.mapped_bytes(),
        0,
        "a released pool must return its physical granules"
    );

    mapper.release();
    for tensor in arenas.k.drain(..).chain(arenas.v.drain(..)) {
        gpu.free_tensor(tensor)?;
    }
    assert_eq!(gpu.vmm_allocation_count(), 0);
    println!("vmm_paged_reclaim_smoke: PASS");
    Ok(())
}
