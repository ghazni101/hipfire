// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Render tail think tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    
    use hipfire_generate::{common::asst_turn_fingerprint, common::normalize_asst_turn_for_fingerprint};
    use hipfire_runtime::prompt_frame::AssistantPrefix;

    #[test]
    fn qwen_jinja_think_tail_primes_reasoning_channel() {
        assert!(render_tail_opens_think("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn speculative_emitter_uses_rendered_think_state() {
        assert!(matches!(
            spec_assistant_prefix(true),
            AssistantPrefix::OpenThink
        ));
        assert!(matches!(
            spec_assistant_prefix(false),
            AssistantPrefix::Plain
        ));
    }

    #[test]
    fn plain_closed_and_user_literal_tails_do_not_prime() {
        assert!(!render_tail_opens_think("<|im_start|>assistant\n"));
        assert!(!render_tail_opens_think(
            "<|im_start|>assistant\n<think>\n</think>\n"
        ));
        assert!(!render_tail_opens_think(
            "<|im_start|>user\nliteral <think><|im_end|>\n<|im_start|>assistant\n"
        ));
    }

    #[test]
    fn assistant_cache_fingerprint_matches_client_visible_content() {
        let raw = "hidden reasoning</think>\n\nvisible answer<|im_end|>";
        let normalized = hipfire_generate::common::normalize_asst_turn_for_fingerprint(raw);
        assert_eq!(normalized, "visible answer");
        assert_eq!(
            hipfire_generate::common::asst_turn_fingerprint(&normalized, &[]),
            hipfire_generate::common::asst_turn_fingerprint("visible answer", &[])
        );
    }

#[test]
fn ifm_thinking_channels_survive_every_chunk_boundary() {
    use hipfire_runtime::emit_text::{ThinkOutputRouter, ThinkRouteEvent};
    for (open, close) in [
        ("<ifm|think>", "</ifm|think>"),
        ("<ifm|think_fast>", "</ifm|think_fast>"),
        ("<ifm|think_faster>", "</ifm|think_faster>"),
    ] {
        let prompt = format!("<|ifm|im_start|>assistant\n{open}\n");
        assert!(render_tail_opens_think(&prompt));
        assert!(!render_tail_opens_think(&format!("{prompt}{close}")));
        let text = format!("reason{close}\n\nanswer{open}more{close}done");
        for split in 0..=text.len() {
            let mut router = ThinkOutputRouter::new(true);
            let mut events = Vec::new();
            router.push_into(&text[..split], &mut events);
            router.push_into(&text[split..], &mut events);
            router.finish_into(&mut events);
            let mut reasoning = String::new();
            let mut content = String::new();
            for event in events {
                match event {
                    ThinkRouteEvent::Reasoning(s) => reasoning.push_str(&s),
                    ThinkRouteEvent::Content(s) => content.push_str(&s),
                }
            }
            assert_eq!(reasoning, "reasonmore", "split={split}");
            assert_eq!(content, "answerdone", "split={split}");
            assert!(!router.in_think());
        }
    }
}

#[test]
fn ifm_ar_wire_routes_answer_and_stops_at_eot() {
    use hipfire_runtime::emit_text::{ThinkOutputRouter, ToolOutputRouter};
    use hipfire_runtime::eos_filter::EosFilter;
    let text = b"reason</ifm|think>Red, blue, and green.<|ifm|im_end|><|ifm|endoftext|>";
    for split in 0..=text.len() {
        let mut config = qwen_ar_eos_filter_config();
        config.stop_at.extend([b"<|ifm|im_end|>".to_vec(), b"<|ifm|endoftext|>".to_vec()]);
        let mut filter = EosFilter::new(config);
        let mut think = ThinkOutputRouter::new(true);
        let mut tools = ToolOutputRouter::disabled();
        let mut visible = String::new();
        let mut wire = Vec::new();
        let mut stopped = false;
        for chunk in [&text[..split], &text[split..]] {
            if qwen_ar_observe_and_route(&mut wire, "ifm", &mut filter, &mut think, &mut tools, chunk, &mut visible).unwrap() {
                stopped = true;
                break;
            }
        }
        qwen_ar_drain_pending_into_router(&mut wire, "ifm", &mut filter, &mut think, &mut tools, &mut visible).unwrap();
        let mut reasoning = String::new();
        let mut answer = String::new();
        for line in String::from_utf8(wire).unwrap().lines() {
            let event: serde_json::Value = serde_json::from_str(line).unwrap();
            match event["type"].as_str().unwrap() {
                "reasoning" => reasoning.push_str(event["text"].as_str().unwrap()),
                "token" => answer.push_str(event["text"].as_str().unwrap()),
                other => panic!("unexpected event {other}"),
            }
        }
        assert!(stopped, "split={split}");
        assert_eq!(reasoning, "reason", "split={split}");
        assert_eq!(answer, "Red, blue, and green.", "split={split}");
    }
}
