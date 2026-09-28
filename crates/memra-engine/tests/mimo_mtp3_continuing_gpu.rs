//! Source-owned MiMo MTP3 depth proposals on one continuing native target.
//!
//! This is a serial correctness gate. A depth proposes one token, the current
//! target logits verify it, and the teacher token is committed before the next
//! depth consumes that token. Neither forward has speculative KV rollback or a
//! multi-row target verifier, so depth matches are not a bundled MTP accept
//! length, throughput result, or serving qualification.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_mtp_forward::{MiMoMtp3Forward, MiMoMtp3Step};
use memra_engine::mimo_mtp_weights::Mtp3Weights;
use memra_engine::mimo_text_forward::MiMoTextStep;
use memra_engine::mimo_text_weights::{MiMoTextWeights, stage_for_layer};
use memra_gguf::source::SafetensorsSource;

type Fail = Box<dyn Error>;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
const HIDDEN: usize = 4_096;
const VOCAB: usize = 152_576;
const DEPTHS: usize = 3;
const TURNS: usize = 2;
const PREFIX: [u32; 3] = [42, 43, 44];
const FOLLOWUP: [u32; 2] = [45, 46];
const TOTAL_TOKENS: usize = PREFIX.len() + FOLLOWUP.len() + TURNS * DEPTHS;

#[derive(Debug, PartialEq, Eq)]
struct CursorReceipt {
    cached_tokens_in: usize,
    new_input_tokens: usize,
    cursor_out: usize,
}

fn cursor_receipt(
    cached_tokens_in: usize,
    cursor_out: usize,
    new_input_tokens: usize,
) -> Result<CursorReceipt, &'static str> {
    if cached_tokens_in.checked_add(new_input_tokens) != Some(cursor_out) {
        return Err("MiMo continuing cursor does not account for new input");
    }
    Ok(CursorReceipt {
        cached_tokens_in,
        new_input_tokens,
        cursor_out,
    })
}

/// MTP depth 0 reads token[j] with target hidden[j]. Depth 1 reads
/// token[j+1] with depth-0 hidden[j]; depth 2 reads token[j+2] with
/// depth-1 hidden[j]. A missing committed input must never be drafted.
fn depth_input_index(
    depth: usize,
    position: usize,
    committed_tokens: usize,
) -> Result<usize, &'static str> {
    if depth >= DEPTHS {
        return Err("MiMo MTP3 depth is outside the pinned source order");
    }
    let index = position
        .checked_add(depth)
        .ok_or("MiMo MTP3 source input index overflowed")?;
    if index >= committed_tokens {
        return Err("MiMo MTP3 source input is not teacher committed");
    }
    Ok(index)
}

fn argmax(logits: &[f32]) -> Result<u32, &'static str> {
    if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
        return Err("MiMo target or MTP3 logits are incomplete");
    }
    let mut best = 0;
    for index in 1..VOCAB {
        if logits[index].total_cmp(&logits[best]).is_gt() {
            best = index;
        }
    }
    Ok(best as u32)
}

fn draft_step(
    draft: &mut MiMoMtp3Forward<'_>,
    depth: usize,
    position: usize,
    tokens: &[u32],
    target_steps: &[MiMoTextStep],
    draft_hidden: &[Vec<Vec<f32>>; DEPTHS],
) -> Result<MiMoMtp3Step, Fail> {
    let input_index = depth_input_index(depth, position, tokens.len())?;
    if draft.position(depth)? != position {
        return Err("MiMo MTP3 depth cursor did not retain its committed prefix".into());
    }
    let parent = if depth == 0 {
        target_steps
            .get(position)
            .map(|step| step.hidden_before_norm.as_slice())
    } else {
        draft_hidden[depth - 1].get(position).map(Vec::as_slice)
    }
    .ok_or("MiMo MTP3 source-ordered parent hidden row is missing")?;
    let step = draft.token(depth, position, tokens[input_index], parent)?;
    if step.depth != depth
        || step.position != position
        || step.hidden_before_norm.len() != HIDDEN
        || draft.position(depth)? != position + 1
    {
        return Err("MiMo MTP3 depth step or resident KV cursor is incomplete".into());
    }
    Ok(step)
}

fn same_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[test]
fn source_order_and_cursor_receipts_reject_uncommitted_rows() {
    assert_eq!(depth_input_index(0, 2, 3), Ok(2));
    assert_eq!(
        depth_input_index(1, 2, 3),
        Err("MiMo MTP3 source input is not teacher committed")
    );
    assert_eq!(depth_input_index(2, 2, 5), Ok(4));
    assert!(depth_input_index(2, 6, 8).is_err());
    assert_eq!(depth_input_index(2, 6, 9), Ok(8));
    assert!(depth_input_index(3, 0, 5).is_err());
    assert!(depth_input_index(2, usize::MAX, 5).is_err());
    assert_eq!(
        cursor_receipt(6, 9, 3),
        Ok(CursorReceipt {
            cached_tokens_in: 6,
            new_input_tokens: 3,
            cursor_out: 9,
        })
    );
    assert!(cursor_receipt(6, 9, 2).is_err());
    assert_eq!(cursor_receipt(6, 11, 5).unwrap().new_input_tokens, 5);
    assert_eq!(cursor_receipt(3, 8, 5).unwrap().cached_tokens_in, 3);
}

#[test]
#[ignore = "requires the pinned MiMo source and a dedicated two-card GPU lane"]
fn native_continuing_mtp3_depths_verify_against_serial_teacher() -> Result<(), Fail> {
    let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
    let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    assert_eq!(cards[0].stream().context().ordinal(), 0);
    assert_eq!(cards[1].stream().context().ordinal(), 1);
    let text = MiMoTextWeights::load(engines, source.clone())?;
    let mtp = Mtp3Weights::load(&cards[0], source.clone())?;
    assert!(Arc::ptr_eq(&mtp.source, &source));
    assert_eq!(mtp.device_ordinal, 0);
    assert_eq!(text.layers.len(), 48);
    assert_eq!([stage_for_layer(0)?, stage_for_layer(23)?], [0, 0]);
    assert_eq!([stage_for_layer(24)?, stage_for_layer(47)?], [1, 1]);
    assert_eq!(text.output_head.ordinal(), 1);
    for (depth, layer) in mtp.layers.iter().enumerate() {
        assert_eq!(layer.depth as usize, depth);
    }
    mtp.check_device(&cards[0])?;
    // draft_forward also rejects a different Arc source owner and audits every
    // text/MTP resident tensor against the two stage devices.
    let mut draft = mtp.draft_forward(&text, &cards[0], &cards[1])?;
    let mut target = text.compressed_text_forward(engines, TOTAL_TOKENS, [FOUR_GIB; 2])?;
    assert!(!target.has_modal_payload());

    let mut tokens = Vec::with_capacity(TOTAL_TOKENS);
    let mut generated = Vec::with_capacity(TOTAL_TOKENS);
    let mut target_steps = Vec::with_capacity(TOTAL_TOKENS);
    let mut draft_hidden: [Vec<Vec<f32>>; DEPTHS] = std::array::from_fn(|_| Vec::new());
    println!("format\tmemra-mimo-mtp3-native-continuing-serial-gate-v1");
    println!("source_owner\tshared_pinned_safetensors");
    println!("placement\ttext_0_23_gpu0_24_47_gpu1_mtp_gpu0_head_gpu1");
    println!("verification\tdepthwise_serial_teacher_before_next_depth");

    for turn in 0..TURNS {
        let fresh_input = if turn == 0 {
            PREFIX.as_slice()
        } else {
            FOLLOWUP.as_slice()
        };
        let cached_before = target.position();
        for &token in fresh_input {
            let position = tokens.len();
            let step = target.token_with_hidden(token)?;
            assert_eq!(step.position, position);
            assert_eq!(target.position(), position + 1);
            assert_eq!(step.hidden_before_norm.len(), HIDDEN);
            tokens.push(token);
            generated.push(false);
            target_steps.push(step);
        }
        let prompt_receipt = cursor_receipt(cached_before, target.position(), fresh_input.len())?;
        if turn > 0 {
            assert!(prompt_receipt.cached_tokens_in >= PREFIX.len() + DEPTHS);
        }
        let root_position = tokens.len() - 1;
        let target_before = target.position();
        let draft_before: [usize; DEPTHS] =
            std::array::from_fn(|depth| draft.position(depth).unwrap());
        assert_eq!(target_before, root_position + 1);
        if turn > 0 {
            assert!(draft_before.into_iter().all(|position| position > 0));
        }
        let mut matches = 0;
        for depth in 0..DEPTHS {
            while draft.position(depth)? < root_position {
                let position = draft.position(depth)?;
                let step = draft_step(
                    &mut draft,
                    depth,
                    position,
                    &tokens,
                    &target_steps,
                    &draft_hidden,
                )?;
                draft_hidden[depth].push(step.hidden_before_norm);
            }
            let input_index = depth_input_index(depth, root_position, tokens.len())?;
            let proposal = draft_step(
                &mut draft,
                depth,
                root_position,
                &tokens,
                &target_steps,
                &draft_hidden,
            )?;
            let proposed = argmax(&proposal.logits)?;
            draft_hidden[depth].push(proposal.hidden_before_norm);

            // The last *committed* target row supplies the teacher's next-token
            // logits. Check the proposal before executing the teacher token.
            let teacher_token = argmax(&target_steps.last().unwrap().logits)?;
            let matched = proposed == teacher_token;
            matches += usize::from(matched);
            println!(
                "proposal\tturn={turn}\tdepth={depth}\tposition={root_position}\tinput_index={input_index}\tproposed={proposed}\tteacher={teacher_token}\tmatch={matched}"
            );
            let before_commit = target.position();
            let committed = target.token_with_hidden(teacher_token)?;
            assert_eq!(committed.position, before_commit);
            assert_eq!(target.position(), before_commit + 1);
            assert_eq!(committed.hidden_before_norm.len(), HIDDEN);
            tokens.push(teacher_token);
            generated.push(true);
            target_steps.push(committed);
        }
        let generated_receipt = cursor_receipt(target_before, target.position(), DEPTHS)?;
        let target_receipt =
            cursor_receipt(cached_before, target.position(), fresh_input.len() + DEPTHS)?;
        let expected_draft_rows = if turn == 0 {
            PREFIX.len()
        } else {
            DEPTHS + FOLLOWUP.len()
        };
        let draft_receipts: [CursorReceipt; DEPTHS] = std::array::from_fn(|depth| {
            cursor_receipt(
                draft_before[depth],
                draft.position(depth).unwrap(),
                expected_draft_rows,
            )
            .unwrap()
        });
        assert_eq!(target.position(), tokens.len());
        assert_eq!(
            generated_receipt.cached_tokens_in,
            prompt_receipt.cursor_out
        );
        assert_eq!(draft_receipts[0], draft_receipts[1]);
        assert_eq!(draft_receipts[1], draft_receipts[2]);
        println!(
            "turn\t{turn}\ttarget_cached_tokens_in={}\ttarget_new_input_tokens={}\ttarget_generated_tokens={}\ttarget_cursor_out={}\tdraft_cached_rows_in={}\tdraft_new_rows={}\tdraft_cursor_out={}\tdepth_matches={matches}/3",
            target_receipt.cached_tokens_in,
            prompt_receipt.new_input_tokens,
            generated_receipt.new_input_tokens,
            target_receipt.cursor_out,
            draft_receipts[0].cached_tokens_in,
            draft_receipts[0].new_input_tokens,
            draft_receipts[0].cursor_out,
        );
    }
    assert_eq!(tokens.len(), TOTAL_TOKENS);
    assert_eq!(generated.len(), TOTAL_TOKENS);
    assert_eq!(target.position(), TOTAL_TOKENS);

    // Separate serial teacher control, replaying the exact committed tape.
    // Compare every target logit and hidden bit, not only its greedy token.
    drop(target);
    let mut control = text.compressed_text_forward(engines, TOTAL_TOKENS, [FOUR_GIB; 2])?;
    for (position, &token) in tokens.iter().enumerate() {
        let step = control.token_with_hidden(token)?;
        let candidate = &target_steps[position];
        assert_eq!(step.position, position);
        assert_eq!(control.position(), position + 1);
        assert!(
            same_bits(&step.logits, &candidate.logits),
            "target logits differ from serial teacher at position {position}"
        );
        assert!(
            same_bits(&step.hidden_before_norm, &candidate.hidden_before_norm),
            "target hidden differs from serial teacher at position {position}"
        );
        if generated.get(position + 1).copied().unwrap_or(false) {
            assert_eq!(
                argmax(&step.logits)?,
                tokens[position + 1],
                "serial teacher chose a different generated token at position {position}"
            );
        }
    }
    println!("serial_teacher_exact_tokens\t{TOTAL_TOKENS}");
    Ok(())
}
