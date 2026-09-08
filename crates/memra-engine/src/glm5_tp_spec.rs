//! CPU-side contracts shared by symmetric TP verify admission and rollback.

/// All fields describe the loaded program, except the explicit experimental door.
pub struct Admission {
    pub enabled: bool,
    pub ranks: usize,
    pub all_layers: bool,
    pub peer_ar: bool,
    pub split_experts: bool,
    pub indexers: bool,
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
        if !self.indexers {
            return Some("TP spec requires DSA indexers on MLA head shards");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn admission() -> Admission {
        Admission {
            enabled: true,
            ranks: 2,
            all_layers: true,
            peer_ar: true,
            split_experts: true,
            indexers: true,
            dflash: true,
            batch: true,
            graphs: false,
        }
    }

    #[test]
    fn admission_refuses_each_missing_precondition() {
        assert_eq!(admission().refusal(), None);
        let arms: [fn(&mut Admission); 10] = [
            |a| a.enabled = false,
            |a| a.ranks = 1,
            |a| a.ranks = 4,
            |a| a.all_layers = false,
            |a| a.peer_ar = false,
            |a| a.split_experts = false,
            |a| a.indexers = false,
            |a| a.dflash = false,
            |a| a.batch = false,
            |a| a.graphs = true,
        ];
        for arm in arms {
            let mut a = admission();
            arm(&mut a);
            assert!(a.refusal().is_some());
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Rank {
        state: u64,
        conv: Vec<u32>,
        kv: usize,
        pools: usize,
    }

    impl Rank {
        fn step(&mut self, token: u32) {
            self.state = self.state.wrapping_mul(31).wrapping_add(u64::from(token));
            self.conv.rotate_left(1);
            *self.conv.last_mut().unwrap() = token;
            self.kv += 1;
            self.pools = self.kv / 4;
        }
    }

    #[test]
    fn both_ranks_restore_every_accepted_prefix_and_continue() {
        for start in [3, 4, 31] {
            let snapshots = [
                Rank {
                    state: 17,
                    conv: vec![1, 2, 3, 4],
                    kv: start,
                    pools: start / 4,
                },
                Rank {
                    state: 91,
                    conv: vec![9, 8, 7, 6],
                    kv: start,
                    pools: start / 4,
                },
            ];
            for t in 1..=7 {
                let rows: Vec<u32> = (10..10 + t as u32).collect();
                for keep in 1..=t {
                    let mut live = snapshots.clone();
                    for rank in &mut live {
                        for &id in &rows {
                            rank.step(id);
                        }
                    }
                    let mut expected = snapshots.clone();
                    for rank in &mut expected {
                        for &id in &rows[..keep] {
                            rank.step(id);
                        }
                    }
                    let decision = Commit::new(start, t, keep).unwrap();
                    let mut restored = Vec::new();
                    decision
                        .restore::<&str>(2, &[t, t], |r, count| {
                            restored.push(r);
                            live[r] = snapshots[r].clone();
                            for &id in &rows[..count] {
                                live[r].step(id);
                            }
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(live, expected, "start={start} t={t} keep={keep}");
                    assert_eq!(decision.end(), live[0].kv);
                    assert_eq!(decision.end(), live[1].kv);
                    assert_eq!(restored, if keep == t { vec![] } else { vec![0, 1] });
                    // A rejected suffix must not contaminate the next round's state.
                    for r in 0..2 {
                        live[r].step(99);
                        expected[r].step(99);
                    }
                    assert_eq!(live, expected);
                    assert_ne!(
                        live[0].state, live[1].state,
                        "root must not overwrite peer state"
                    );
                }
            }
        }
    }

    #[test]
    fn torn_checkpoint_refuses_before_either_rank_changes() {
        let commit = Commit::new(19, 7, 3).unwrap();
        for counts in [vec![], vec![7], vec![7, 6], vec![7, 7, 7]] {
            let mut touched = Vec::new();
            let result = commit.restore::<&str>(2, &counts, |r, _| {
                touched.push(r);
                Ok(())
            });
            assert!(result.is_err());
            assert!(touched.is_empty());
        }
    }

    #[test]
    fn invalid_decision_and_rank_failure_never_report_success() {
        for (pos, rows, keep) in [(0, 0, 0), (0, 7, 0), (0, 7, 8), (usize::MAX, 7, 1)] {
            assert!(Commit::new(pos, rows, keep).is_err());
        }
        let commit = Commit::new(0, 7, 3).unwrap();
        let mut calls = Vec::new();
        let result = commit.restore::<&str>(2, &[7, 7], |r, _| {
            calls.push(r);
            if r == 1 { Err("peer failed") } else { Ok(()) }
        });
        assert_eq!(calls, vec![0, 1]);
        assert_eq!(result, Err("peer failed"));
    }
}
