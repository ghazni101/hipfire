// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 2 smoke test for [`rdna_compute::vmm_page_pool::VmmPageMapper`].
//!
//! Proves on hardware:
//! - initial load commits no physical KV (VA-only)
//! - a resident page maps its whole granule in every K/V arena
//! - pages at non-adjacent granule offsets read/write at the expected offsets
//! - freeing the last page in a granule unmaps it and reclaims the bytes
//! - remapping a freed page commits fresh memory
//! - a budget refusal is an error, not a silent over-commit
//!
//! Run:
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p rdna-compute --example vmm_page_mapper_smoke

use rdna_compute::page_pool::{PageState, PAGE_TOKENS};
use rdna_compute::vmm_page_pool::{VmmPageArenas, VmmPageMapper};
use rdna_compute::Gpu;

const PER_POS: usize = 1088;
const N_PAGES: usize = 4096;
const N_LAYERS: usize = 1;

fn dev() -> i32 {
    std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0)
}

fn states(resident: &[usize]) -> Vec<PageState> {
    let mut values = vec![PageState::Free; N_PAGES];
    for &page in resident {
        values[page] = PageState::Private;
    }
    values
}

fn write_read(
    gpu: &mut Gpu,
    mapper: &VmmPageMapper,
    arenas: &VmmPageArenas,
    page: usize,
    pattern: u8,
) {
    let offset = mapper.page_offset(page as u32).expect("page offset");
    let bytes = vec![pattern; 4096];
    gpu.hip
        .memcpy_htod_offset(&arenas.k[0].buf, offset, &bytes)
        .expect("write page");
    gpu.hip
        .memcpy_htod_offset(&arenas.v[0].buf, offset, &bytes)
        .expect("write V page");
    gpu.hip.device_synchronize().expect("sync");
    let mut readback = vec![0u8; bytes.len()];
    gpu.hip
        .memcpy_dtoh_at(&mut readback, &arenas.k[0].buf, offset)
        .expect("read page");
    assert_eq!(readback, bytes, "K page {page} readback");
    gpu.hip
        .memcpy_dtoh_at(&mut readback, &arenas.v[0].buf, offset)
        .expect("read V page");
    assert_eq!(readback, bytes, "V page {page} readback");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init_with_device(dev())?;
    let (mut mapper, mut arenas) =
        VmmPageMapper::new(&mut gpu, N_LAYERS, N_PAGES, PAGE_TOKENS, PER_POS, 0, 0)?;
    let layout = mapper.page_layout();
    assert!(layout.pages_per_granule > 0);
    assert!(layout.granule_bytes >= layout.page_bytes);
    assert!(
        layout.pages_per_granule as usize * layout.page_bytes <= layout.granule_bytes,
        "pages must fit inside a granule"
    );
    assert_eq!(mapper.mapped_bytes(), 0, "VA-only load must map no KV");
    assert_eq!(
        layout.n_granules,
        N_PAGES.div_ceil(layout.pages_per_granule as usize)
    );
    println!(
        "vmm_page_mapper_smoke: granule={} page={} P={} granules={}",
        layout.granule_bytes, layout.page_bytes, layout.pages_per_granule, layout.n_granules
    );

    // No page resident: nothing maps, and a table naming a page is refused.
    mapper.sync(&mut gpu, &states(&[]))?;
    assert_eq!(mapper.mapped_bytes(), 0);
    assert!(mapper.check_table(&[0], 1).is_err());

    // Two non-adjacent granules: page 0 and the first page of the next
    // granule. (The host's 2 MiB granule packs 15 pages; a 4 KiB granule
    // cannot hold one page and gets the rounded-up page as its granule.)
    let other = (layout.pages_per_granule as usize).min(N_PAGES - 1);
    assert_ne!(other / layout.pages_per_granule as usize, 0);
    mapper.sync(&mut gpu, &states(&[0, other]))?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    let granule_cost = layout.granule_bytes * 2 * N_LAYERS;
    assert_eq!(mapper.mapped_bytes(), 2 * granule_cost);
    let mut table = vec![0u32; 2];
    table[0] = 0;
    table[1] = other as u32;
    mapper.check_table(&table, 2)?;
    write_read(&mut gpu, &mapper, &arenas, 0, 0x11);
    write_read(&mut gpu, &mapper, &arenas, other, 0x99);

    // Release one page: its granule stays resident for the other page.
    mapper.sync(&mut gpu, &states(&[other]))?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(mapper.mapped_bytes(), granule_cost);
    assert!(mapper.check_table(&[0], 1).is_err());
    // Release the last page: all physical KV is reclaimed.
    mapper.sync(&mut gpu, &states(&[]))?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(
        mapper.mapped_bytes(),
        0,
        "released granules must be unmapped"
    );

    // Re-map a freed page and write through the fresh physical allocation.
    mapper.sync(&mut gpu, &states(&[other]))?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    write_read(&mut gpu, &mapper, &arenas, other, 0x42);
    assert_eq!(mapper.mapped_bytes(), granule_cost);
    mapper.release();
    for t in arenas.k.drain(..).chain(arenas.v.drain(..)) {
        gpu.free_tensor(t)?;
    }
    assert_eq!(gpu.vmm_allocation_count(), 0, "all owners released");

    // A budget for only one granule group refuses the second group. The
    // first newly-mapped group is rolled back, leaving the mapper usable for
    // an explicit release rather than half-updated.
    let (mut tight, mut tight_arenas) = VmmPageMapper::new(
        &mut gpu,
        N_LAYERS,
        N_PAGES,
        PAGE_TOKENS,
        PER_POS,
        0,
        granule_cost,
    )?;
    let err = tight
        .sync(&mut gpu, &states(&[0, other]))
        .expect_err("two granule groups must exceed a one-group budget");
    assert!(err.contains("budget"), "unexpected budget error: {err}");
    tight.release();
    for t in tight_arenas.k.drain(..).chain(tight_arenas.v.drain(..)) {
        gpu.free_tensor(t)?;
    }

    println!("vmm_page_mapper_smoke: PASS");
    Ok(())
}
