//! CPU-side contracts shared by symmetric TP verify admission and rollback.

/// All fields describe the loaded program, except the explicit experimental door.
pub struct Admission {
    pub enabled: bool,
    pub ranks: usize,
    pub all_layers: bool,
    pub peer_ar: bool,
    pub split_experts: bool,
    pub dflash: bool,
    pub batch: bool,
    pub graphs: bool,
}

impl Admission {
    pub fn refusal(&self) -> Option<&'static str> {
        if !self.enabled {
            return Some("MEMRA_GLM5_SPEC_TP is off");
        }
        if self.ranks != 2 || !self.all_layers || !self.peer_ar {
            return Some("TP spec requires full symmetric TP2 with peer all-reduce");
        }
        if !self.split_experts {
            return Some("TP spec requires split experts and peer glue");
        }
        if !self.dflash {
            return Some("TP spec requires the DFlash2 drafter");
        }
        if !self.batch {
            return Some("TP spec requires MEMRA_GLM5_VERIFY_BATCH");
        }
        if self.graphs {
            return Some("TP spec verify is eager only; disable MEMRA_GLM5_VERIFY_GRAPH");
        }
        None
    }
}

/// One root decision applies to every rank. The anchor is always kept; the bonus
/// token belongs to the next round and does not advance this round's KV cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Commit {
    pub pos: usize,
    pub rows: usize,
    pub keep: usize,
}

impl Commit {
    pub fn new(pos: usize, rows: usize, keep: usize) -> Result<Self, &'static str> {
        if keep == 0 || keep > rows {
            return Err("TP verify keep outside 1..=rows");
        }
        pos.checked_add(rows).ok_or("TP verify cursor overflow")?;
        Ok(Self { pos, rows, keep })
    }

    pub fn end(self) -> usize {
        self.pos + self.keep
    }

    /// Validate every stash before invoking any rank. A torn checkpoint must not
    /// partially restore root and then discover a missing peer snapshot.
    pub fn restore<E: From<&'static str>>(
        self,
        ranks: usize,
        stash_rows: &[usize],
        mut restore: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        if ranks < 2 || stash_rows.len() != ranks || stash_rows.iter().any(|&t| t != self.rows) {
            return Err("TP verify missing or mismatched rank checkpoint".into());
        }
        if self.keep != self.rows {
            for rank in 0..ranks {
                restore(rank, self.keep)?;
            }
        }
        Ok(())
    }
}
