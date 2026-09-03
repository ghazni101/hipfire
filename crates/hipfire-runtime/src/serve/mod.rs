// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Concurrent serve: several agents served at once by the multi-slot engine.
//
// `hipfire serve` already exposes an OpenAI-compatible /v1/chat/completions
// over HTTP, and HTTP is already concurrent. What serialises requests today is
// behind it: one `Engine` (a single daemon process) guarded by a
// `Mutex<ServeRuntime>`. `SlotEngine` is an alternative backend that lives
// OUTSIDE that mutex — which is the entire point — so requests overlap.

// The engine LOOP lives in `hipfire-arch-qwen35::serve_engine`, not here:
// it needs `forward_batch_slots_graphed`, and that crate already depends on
// this one, so putting the loop here would be a dependency cycle. This module
// owns the protocol types both sides share — the same layering reason
// `swap::snapshot` takes an opaque buffer slice instead of a `DeltaNetState`.

use std::sync::mpsc::Sender;

/// Visual payload for a VL request.
///
/// Image decode is CPU-side (daemon). `vision_forward` runs on the slot
/// engine thread, which exclusively owns the GPU and VisionWeights.
/// The engine splices the resulting embeddings at `<|image_pad|>` positions
/// during a per-token prefill that uses `forward_scratch_embed_mrope`.
///
/// `mrope_positions` carries the 3D (t, h, w) position for every prompt token,
/// already offset by `base` so values are absolute rope phases. `rope_delta`
/// is added to the running sequence length for decode-step positions.
pub struct VisualData {
    /// Vision-tower patch tensor from `extract_patches` (CPU).
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Number of post-merge visual tokens to splice at image_pad positions.
    pub n_visual_tokens: usize,
    /// 3D M-RoPE positions for every prompt token, offset by base.
    pub mrope_positions: Vec<[i32; 3]>,
    /// rope_delta for decode positions past the prompt.
    pub rope_delta: i32,
}

/// One request handed to the engine.
///
/// `reply` is the client's own channel: the engine streams this request's
/// events down it and nothing else. A dropped receiver is how the engine
/// learns the client is gone.
pub struct SubmitRequest {
    /// Full cold render of the conversation. Used when nothing is reusable.
    pub prompt_tokens: Vec<u32>,
    /// Hashes of the conversation's user turns, in order. Identity for
    /// continuation matching — see `Session::convo`.
    pub convo: Vec<u64>,
    pub continuation: Continuation,
    pub max_tokens: usize,
    /// 0.0 = greedy. Per-request; the engine installs it on the slot.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k.
    pub top_k: i32,
    pub seed: u32,
    /// Recency window in recent tokens for the penalties below. 0 disables
    /// token penalties regardless of the penalty values.
    pub repeat_window: usize,
    /// Multiplicative recency-weighted repeat penalty; 1.0 = off.
    pub repeat_penalty: f32,
    /// OpenAI flat presence penalty; 0.0 = off. This is the mechanism that
    /// suppresses block-level repetition loops on long generations.
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (scaled by in-window count); 0.0 = off.
    pub frequency_penalty: f32,
    /// min-p cutoff; 0.0 = off.
    pub min_p: f32,
    /// Visual embeddings + M-RoPE for VL requests. None for text-only.
    pub visual_data: Option<VisualData>,
    pub reply: Sender<Event>,
}

/// How this turn extends what the engine already holds.
///
/// The two reuse shapes are NOT interchangeable, and untyped tokens plus a
/// convo hash cannot tell them apart: a new user turn and a tool-result
/// iteration both arrive as "a suffix to append", but a tool-result suffix
/// pinned onto a session that never emitted those calls feeds the model a
/// `<tool_response>` for a call it did not make, and a user-turn suffix pinned
/// onto a session that already served that turn duplicates it in the KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Continuation {
    /// Nothing to append: the engine matches on `prompt_tokens` alone.
    Cold,
    /// A new user turn, from `prompt_frame::continuation_suffix`. Matched
    /// against a session holding exactly the preceding user turns; the tokens
    /// are APPENDED to its stored tokens, so the result is a strict extension
    /// of the KV rather than a re-render of it.
    UserTurn(Vec<u32>),
    /// Tool results answering the tool calls `session` emitted on its previous
    /// turn, from `prompt_frame::continuation_suffix_tool_results`. The caller
    /// resolved `session` from the tool-call ids it handed the client, so the
    /// engine only re-checks that the session still holds this conversation.
    ToolResults { tokens: Vec<u32>, session: u64 },
}

impl Continuation {
    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Cold => &[],
            Self::UserTurn(tokens) => tokens,
            Self::ToolResults { tokens, .. } => tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoneReason {
    Eos,
    MaxTokens,
    /// The client's receiver was dropped. The session is closed and its slot
    /// freed rather than generating into a void.
    ClientGone,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// `reused` — prompt tokens served from the session's existing KV;
    /// `prefill` — tokens actually prefilled this turn. Together they are the
    /// client-visible prompt accounting (OpenAI `usage.prompt_tokens` =
    /// reused + prefill, `cached_tokens` = reused).
    Accepted {
        session: u64,
        reused: usize,
        prefill: usize,
    },
    Token {
        id: u32,
    },
    /// `generated` — tokens delivered to the client (terminators excluded).
    Done {
        reason: DoneReason,
        generated: usize,
    },
    Rejected {
        reason: String,
    },
}

/// Post an event to a client, reporting a vanished receiver as an error.
///
/// The engine must be able to see this: a request whose client has gone still
/// holds a slot, and on a 4-slot box that is 25% of capacity generating output
/// nobody will read.
pub fn send_event(tx: &Sender<Event>, e: Event) -> Result<(), String> {
    tx.send(e)
        .map_err(|_| "client receiver dropped".to_string())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub admitted: usize,
    pub rejected: usize,
    pub evictions: usize,
    pub restores: usize,
    /// Continuations served from a session's existing KV. Counted separately
    /// from `restores`: a hit on a still-resident session restores nothing, and
    /// conflating the two makes a gate that never restores look like it did.
    pub prefix_hits: usize,
}

impl EngineStats {
    pub fn note_admitted(&mut self) {
        self.admitted += 1;
    }
    pub fn note_rejected(&mut self) {
        self.rejected += 1;
    }
    pub fn note_eviction(&mut self) {
        self.evictions += 1;
    }
    pub fn note_restore(&mut self) {
        self.restores += 1;
    }
    pub fn note_prefix_hit(&mut self) {
        self.prefix_hits += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn a_request_whose_client_vanished_is_detected_before_work_is_done() {
        let (tx, rx) = channel::<Event>();
        drop(rx);
        assert!(
            send_event(&tx, Event::Token { id: 1 }).is_err(),
            "a dropped receiver must be visible so the engine can free the slot"
        );
    }

    #[test]
    fn a_live_client_receives_its_events_in_order() {
        let (tx, rx) = channel::<Event>();
        send_event(
            &tx,
            Event::Accepted {
                session: 4,
                reused: 10,
                prefill: 2,
            },
        )
        .unwrap();
        send_event(&tx, Event::Token { id: 7 }).unwrap();
        send_event(
            &tx,
            Event::Done {
                reason: DoneReason::Eos,
                generated: 1,
            },
        )
        .unwrap();
        assert_eq!(
            rx.recv().unwrap(),
            Event::Accepted {
                session: 4,
                reused: 10,
                prefill: 2
            }
        );
        assert_eq!(rx.recv().unwrap(), Event::Token { id: 7 });
        assert_eq!(
            rx.recv().unwrap(),
            Event::Done {
                reason: DoneReason::Eos,
                generated: 1
            }
        );
    }

    #[test]
    fn stats_start_at_zero_and_count_each_outcome() {
        let mut s = EngineStats::default();
        s.note_admitted();
        s.note_admitted();
        s.note_rejected();
        assert_eq!(s.admitted, 2);
        assert_eq!(s.rejected, 1);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.restores, 0);
    }
}
