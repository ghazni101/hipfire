// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Scheduler — decides what goes into each step's SlotBatch.
//
// Deliberately minimal: round-robin, chunked prefill mixed with decode, no
// preemption or priorities. Batching wins ~1.36x at 8 slots and recovers
// under a tenth of the available bandwidth headroom, so contorting the
// scheduler to maximise batch width is not where the remaining performance
// is. Correctness and not blocking other slots matter more.
//
// The one property that matters: a long prompt is chunked to `chunk_size`
// tokens per step so it cannot block other slots (in particular decoding
// slots) for its whole duration. See `prefill_and_decode_mix_in_one_batch`.

use crate::slot_batch::SlotBatch;
use rdna_compute::slot_pool::SlotId;

pub struct Scheduler {
    pub chunk_size: usize,
    /// Serve VL slots on the legacy sequential per-token M-RoPE path instead
    /// of the batched paged-capable one. Opt-in via HIPFIRE_VL_SEQUENTIAL=1;
    /// the batched path is the default (and the only one compatible with a
    /// paged pool).
    pub vl_sequential: bool,
}

/// One slot's outstanding work as seen by the scheduler.
pub struct PendingWork {
    pub slot: SlotId,
    /// Prompt tokens not yet fed through prefill. Drained front-to-back as
    /// chunks are taken.
    pub remaining_prompt: Vec<u32>,
    /// Next absolute position this slot will occupy.
    pub next_pos: usize,
    /// Once prefill is complete, the slot decodes one token per step.
    pub decoding: bool,
    /// VL state. When present, this slot is served by the sequential
    /// M-RoPE path (vision_forward + per-token prefill/decode) and MUST
    /// be skipped by the batched 1D-RoPE scheduler.
    pub vl_prefill: Option<VlPrefill>,
    /// MTP state. When true and still prefilling, this slot's prompt chunks
    /// flow through the batched scheduler (the MTP head's KV is filled
    /// post-forward). Once decoding, the slot is owned by the MTP
    /// draft/verify cycle and MUST be skipped by the batched scheduler.
    pub mtp_active: bool,
    /// Rolling adaptive-retire window: decode cycles spent under MTP and
    /// tokens committed across them. When the window's mean advance stays
    /// below `MTP_RETIRE_MIN_ADVANCE`, the engine retires the slot to plain
    /// AR decode (a spec cycle costs ~2x an AR step, so a head that cannot
    /// beat that loses wall-clock — the genre-conditional trap).
    pub mtp_cycles: usize,
    pub mtp_committed: usize,
    /// Consecutive retire-windows whose mean advance fell below the line.
    pub mtp_retire_fails: usize,
}

/// Visual + M-RoPE data for one slot's VL request.
///
/// On the default batched path the scheduler splices vision embeddings (via
/// per-row `SlotBatch::ext_emb` indices) and M-RoPE phases (via
/// `SlotBatch::pos3`) into the chunked batch like any other slot. The
/// sequential fallback (`vl_sequential`) instead owns the slot for the whole
/// request and prefills/decodes per token.
pub struct VlPrefill {
    /// Vision-tower patches; consumed once by `vision_forward`.
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Post-merge visual token count.
    pub n_visual_tokens: usize,
    /// Filled by `vision_forward` (`n_visual_tokens * dim` floats).
    pub embeddings: Vec<f32>,
    /// Model hidden dim (embedding stride).
    pub dim: usize,
    /// Index into `embeddings` for the next visual token to splice.
    pub visual_idx: usize,
    /// Prompt token id that carries one visual embedding per occurrence.
    pub image_pad_id: u32,
    /// 3D M-RoPE positions for every prompt token, offset by base.
    pub mrope_positions: Vec<[i32; 3]>,
    /// rope_delta for decode positions past the prompt.
    pub rope_delta: i32,
    /// Base sequence position for this request's prefill.
    pub base: usize,
}

impl VlPrefill {
    /// Absolute rope phases for position `pos`: this request's prompt table
    /// while in range, `pos + rope_delta` beyond it. Same rule as
    /// `MropeCtx::pos3` on the sequential path.
    pub fn pos3(&self, pos: usize) -> [i32; 3] {
        match pos
            .checked_sub(self.base)
            .and_then(|i| self.mrope_positions.get(i))
        {
            Some(p) => *p,
            None => [pos as i32 + self.rope_delta; 3],
        }
    }
}

impl Scheduler {
    /// Build the next step's `SlotBatch`, taking `min(chunk_size,
    /// remaining_prompt.len())` tokens from each prefilling slot and one
    /// token from each decoding slot, in slot order. Advances `next_pos`
    /// (and drains `remaining_prompt`) for whatever it takes.
    pub fn next_batch(&mut self, work: &mut [PendingWork]) -> SlotBatch {
        // A step carrying VL rows runs the batched M-RoPE kernel for EVERY
        // row (text rows take [p, p, p], bit-identical to 1D RoPE), so the
        // per-row pos3/ext side arrays exist only when some slot is VL.
        let any_vl = work
            .iter()
            .any(|w| w.vl_prefill.is_some() && !self.vl_sequential);
        let mut b = SlotBatch::default();
        for w in work.iter_mut() {
            // Sequential-mode VL slots are owned by the per-token M-RoPE
            // path. MTP slots prefill through this batched path like any
            // other slot (their head-KV fill runs post-forward from x_batch);
            // once decoding, the MTP verify rows are injected by the caller
            // instead, so skip them here.
            if (w.vl_prefill.is_some() && self.vl_sequential)
                || (w.mtp_active && w.decoding)
            {
                b.m_per_slot.push(0);
                continue;
            }
            // Batched-mode VL slots wait until the engine has run
            // vision_forward and uploaded their embedding matrix — building
            // rows before that would embed garbage for the image pads.
            let vl_waiting = w
                .vl_prefill
                .as_ref()
                .is_some_and(|vl| vl.embeddings.is_empty() && vl.n_visual_tokens > 0);
            if vl_waiting {
                b.m_per_slot.push(0);
                continue;
            }
            let take = if w.decoding {
                w.remaining_prompt.len().min(1)
            } else {
                w.remaining_prompt.len().min(self.chunk_size)
            };
            let toks: Vec<u32> = w.remaining_prompt.drain(..take).collect();
            let start_pos = w.next_pos;
            b.m_per_slot.push(toks.len());
            for (i, t) in toks.iter().enumerate() {
                let pos = start_pos + i;
                b.tokens.push(*t);
                b.positions.push(pos as i32);
                b.row_slot.push(w.slot.0 as i32);
                if !any_vl {
                    continue;
                }
                match w.vl_prefill.as_mut() {
                    Some(vl) => {
                        b.pos3.push(vl.pos3(pos));
                        // Image pads splice visual embeddings in prompt
                        // order; everything else uses the token table.
                        if *t == vl.image_pad_id && vl.visual_idx < vl.n_visual_tokens {
                            b.ext_emb.push(vl.visual_idx as i32);
                            vl.visual_idx += 1;
                        } else {
                            b.ext_emb.push(-1);
                        }
                    }
                    None => {
                        b.pos3.push([pos as i32; 3]);
                        b.ext_emb.push(-1);
                    }
                }
            }
            w.next_pos += toks.len();
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    fn prompt(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn a_long_prompt_is_chunked_not_run_whole() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(1000),
            next_pos: 0,
            decoding: false,
            vl_prefill: None, mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(
            b.total_rows(),
            256,
            "must take one chunk, not the whole prompt"
        );
        assert_eq!(work[0].remaining_prompt.len(), 744);
        assert_eq!(work[0].next_pos, 256);
    }

    #[test]
    fn prefill_and_decode_mix_in_one_batch() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(300),
                next_pos: 0,
                decoding: false,
                vl_prefill: None, mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: vec![42],
                next_pos: 10,
                decoding: true,
                vl_prefill: None, mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
            },
        ];
        let b = s.next_batch(&mut work);
        assert_eq!(
            b.m_per_slot,
            vec![256, 1],
            "a prefilling slot must not block a decoding one"
        );
    }

    #[test]
    fn a_prompt_shorter_than_a_chunk_completes_in_one_batch() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(10),
            next_pos: 0,
            decoding: false,
            vl_prefill: None, mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(b.total_rows(), 10);
        assert!(work[0].remaining_prompt.is_empty());
    }

    #[test]
    fn an_idle_slot_contributes_nothing() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: vec![],
            next_pos: 0,
            decoding: false,
            vl_prefill: None, mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
        }];
        let b = s.next_batch(&mut work);
        assert!(b.is_empty());
    }

    fn vl_state(n_visual: usize, n_prompt: usize) -> VlPrefill {
        VlPrefill {
            patches: vec![],
            grid_h: 1,
            grid_w: 1,
            n_visual_tokens: n_visual,
            embeddings: vec![0.0; n_visual * 4],
            dim: 4,
            visual_idx: 0,
            image_pad_id: 42,
            // Phases for the prompt: text rows [p,p,p]; an "image" row at
            // position 4 stretched on h/w (values arbitrary but distinct
            // from the 1D phase, to prove the table wins).
            mrope_positions: (0..n_prompt)
                .map(|p| {
                    if p == 4 {
                        [4, 9, 9]
                    } else {
                        [p as i32, p as i32, p as i32]
                    }
                })
                .collect(),
            rope_delta: 7,
            base: 0,
        }
    }

    #[test]
    fn vl_slots_waiting_on_vision_forward_are_skipped() {
        // Embeddings not yet produced: the slot must not enter the batch
        // (image pads would embed garbage), but the decoding slot next to it
        // is unaffected.
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(10),
                next_pos: 0,
                decoding: false,
                vl_prefill: Some(VlPrefill {
                    embeddings: vec![],
                    ..vl_state(1, 10)
                }),
                mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: vec![7],
                next_pos: 3,
                decoding: true,
                vl_prefill: None, mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
            },
        ];
        let b = s.next_batch(&mut work);
        assert_eq!(b.m_per_slot, vec![0, 1], "unprepared VL slots must wait");
        assert_eq!(work[0].remaining_prompt.len(), 10);
        assert_eq!(work[0].next_pos, 0);
        assert!(work[1].remaining_prompt.is_empty());
    }

    #[test]
    fn batched_vl_rows_carry_mrope_phases() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        // Prompt where position 4 is an image pad.
        let mut toks = prompt(10);
        toks[4] = 42;
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: toks,
            next_pos: 0,
            decoding: false,
            vl_prefill: Some(vl_state(1, 10)),
            mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(b.total_rows(), 10);
        assert_eq!(b.pos3.len(), 10, "one phase triplet per row");
        assert_eq!(b.pos3[4], [4, 9, 9], "the image row reads the table");
        assert_eq!(b.pos3[0], [0, 0, 0], "text rows stay at [p, p, p]");
        assert_eq!(b.ext_emb, vec![-1, -1, -1, -1, 0, -1, -1, -1, -1, -1]);
        assert_eq!(work[0].vl_prefill.as_ref().unwrap().visual_idx, 1);
    }

    #[test]
    fn vl_decode_rows_use_pos_plus_rope_delta() {
        let mut s = Scheduler { chunk_size: 4, vl_sequential: false };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: vec![9],
            next_pos: 12,
            decoding: true,
            vl_prefill: Some(vl_state(2, 4)),
            mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(b.total_rows(), 1);
        // Position 12 is past the 4-row table: [p + rope_delta; 3].
        assert_eq!(b.pos3[0], [12 + 7, 12 + 7, 12 + 7]);
        assert_eq!(b.ext_emb[0], -1, "decode rows never splice embeddings");
    }

    #[test]
    fn sequential_vl_slots_still_bypass_the_batch() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: true };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(10),
            next_pos: 0,
            decoding: false,
            vl_prefill: Some(vl_state(1, 10)),
            mtp_active: false, mtp_cycles: 0, mtp_committed: 0, mtp_retire_fails: 0,
        }];
        let b = s.next_batch(&mut work);
        assert!(b.is_empty(), "sequential mode owns the slot");
        assert!(b.pos3.is_empty(), "no VL rows implies no side arrays");
        assert_eq!(work[0].remaining_prompt.len(), 10);
    }

    #[test]
    fn mtp_prefill_slots_batch_but_decoding_mtp_slots_are_skipped() {
        let mut s = Scheduler { chunk_size: 256, vl_sequential: false };
        let mut work = vec![
            PendingWork {
                // MTP slot still prefilling: its prompt chunk flows through
                // the batched path (the head-KV fill runs post-forward).
                slot: SlotId(0),
                remaining_prompt: prompt(10),
                next_pos: 0,
                decoding: false,
                vl_prefill: None,
                mtp_active: true,
                mtp_cycles: 0,
                mtp_committed: 0,
                mtp_retire_fails: 0,
            },
            PendingWork {
                // MTP slot decoding: owned by the draft/verify cycle; its
                // verify rows are injected by the engine, not the scheduler.
                slot: SlotId(1),
                remaining_prompt: vec![7],
                next_pos: 3,
                decoding: true,
                vl_prefill: None,
                mtp_active: true,
                mtp_cycles: 0,
                mtp_committed: 0,
                mtp_retire_fails: 0,
            },
        ];
        let b = s.next_batch(&mut work);
        assert_eq!(
            b.m_per_slot,
            vec![10, 0],
            "prefilling MTP slots batch; decoding MTP slots are injected separately"
        );
        assert!(work[0].remaining_prompt.is_empty());
        assert_eq!(work[0].next_pos, 10);
        assert_eq!(work[1].remaining_prompt, vec![7], "seed stays put");
        assert_eq!(work[1].next_pos, 3);
    }
}
