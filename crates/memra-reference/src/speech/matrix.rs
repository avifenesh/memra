//! Single-threaded native matrix product for the speech reference executor.
//! Fixed four-bank FP32 FMA order; AVX2 vectorizes independent rows, not the reduction.
//!
//! Rows are processed in cache-resident groups. Grouping changes only which order the
//! independent (row, output) dot products are visited and where their inputs are staged;
//! every dot product keeps the same ascending-k four-bank accumulation, so the result is
//! bit-identical to the ungrouped traversal. `grouping_is_bit_identical` pins that.

/// Target bytes for one packed row group. The group holds `input * group` f32 values and is
/// swept once per output pair, so it must stay in L2 while the weight rows stream past it.
const GROUP_BYTES: usize = 768 * 1024;

/// Worker threads for one matrix product. Default one: every committed speech receipt was
/// measured single-threaded, and threading must never be the reason a number moves. Each
/// worker owns a disjoint block of output columns and accumulates it in the same ascending-k
/// four-bank order, so any thread count returns the same bits.
fn threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("MEMRA_SPEECH_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 64)
    })
}

/// Below this many weight elements a product finishes faster than a hand-off costs.
const MIN_THREADED_WORK: usize = 1 << 16;

struct Job {
    x: (usize, usize),
    w: (usize, usize),
    y: (usize, usize),
    stride: usize,
    input: usize,
    done: std::sync::mpsc::Sender<()>,
}

/// Persistent workers. A pool that spawns per product spends more on hand-off than on
/// arithmetic: the decoder issues tens of thousands of small products per clip.
struct Pool {
    workers: Vec<std::sync::mpsc::Sender<Job>>,
}

impl Pool {
    fn new(workers: usize) -> Self {
        let mut senders = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            std::thread::spawn(move || {
                for job in rx {
                    // SAFETY: `run` blocks until every worker acknowledges, so the slices
                    // outlive the job, and each job owns a disjoint block of output columns.
                    unsafe {
                        let x = std::slice::from_raw_parts(job.x.0 as *const f32, job.x.1);
                        let w = std::slice::from_raw_parts(job.w.0 as *const f32, job.w.1);
                        let y = std::slice::from_raw_parts_mut(job.y.0 as *mut f32, job.y.1);
                        block(x, w, y, job.stride, job.input, w.len() / job.input);
                    }
                    let _ = job.done.send(());
                }
            });
            senders.push(tx);
        }
        Self { workers: senders }
    }

    fn run(
        &self,
        x: &[f32],
        w: &[f32],
        y: &mut [f32],
        stride: usize,
        input: usize,
        columns: usize,
    ) {
        let (done, acks) = std::sync::mpsc::channel();
        let mut issued = 0usize;
        let mut worker = 0usize;
        let mut rest = y;
        let mut offset = 0usize;
        while offset < w.len() {
            let wc = &w[offset..(offset + columns * input).min(w.len())];
            let (yc, tail) = rest.split_at_mut(wc.len() / input * stride);
            rest = tail;
            offset += wc.len();
            if worker == self.workers.len() {
                // More blocks than workers: finish the tail here rather than queueing behind.
                block(x, wc, yc, stride, input, wc.len() / input);
                continue;
            }
            let job = Job {
                x: (x.as_ptr() as usize, x.len()),
                w: (wc.as_ptr() as usize, wc.len()),
                y: (yc.as_mut_ptr() as usize, yc.len()),
                stride,
                input,
                done: done.clone(),
            };
            self.workers[worker].send(job).expect("speech worker alive");
            worker += 1;
            issued += 1;
        }
        for _ in 0..issued {
            acks.recv().expect("speech worker acknowledgement");
        }
    }
}

fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Pool::new(threads() - 1))
}

fn row_group(input: usize, stride: usize) -> usize {
    let rows = (GROUP_BYTES / (input.max(1) * 4)) / 8 * 8;
    rows.clamp(8, stride.max(8))
}

pub(super) fn linear(x: &[f32], w: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * input);
    assert_eq!(w.len(), output * input);
    let stride = rows.div_ceil(8) * 8;
    let group = row_group(input, stride);
    let mut xg = vec![0.0f32; input * group];
    let mut yg = vec![0.0f32; output * group];
    let mut y = vec![0.0f32; rows * output];
    let mut start = 0;
    while start < stride {
        let take = group.min(stride - start);
        // Pack this group's rows as k-major lanes. Rows past the real input are zero, which
        // is what the padded lanes carried before; their results are never read back.
        for lane in 0..take {
            let row = start + lane;
            if row < rows {
                for k in 0..input {
                    xg[k * take + lane] = x[row * input + k];
                }
            } else {
                for k in 0..input {
                    xg[k * take + lane] = 0.0;
                }
            }
        }
        let xp = &xg[..input * take];
        let yp = &mut yg[..output * take];
        let workers = threads()
            .min(output.div_ceil(2))
            .min(input * output / MIN_THREADED_WORK + 1)
            .max(1);
        if workers == 1 {
            block(xp, w, yp, take, input, output);
        } else {
            let columns = output.div_ceil(workers).next_multiple_of(2);
            pool().run(xp, w, yp, take, input, columns);
        }
        for lane in 0..take {
            let row = start + lane;
            if row >= rows {
                break;
            }
            for col in 0..output {
                y[row * output + col] = yp[col * take + lane];
            }
        }
        start += take;
    }
    y
}

fn block(x: &[f32], w: &[f32], y: &mut [f32], stride: usize, input: usize, output: usize) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: features checked above; the packed group is a whole number of eight-row lanes.
        unsafe {
            avx2(x, w, y, stride, input, output);
        }
        return;
    }
    scalar(x, w, y, stride, input, output);
}

fn scalar(x: &[f32], w: &[f32], y: &mut [f32], stride: usize, input: usize, output: usize) {
    for out in 0..output {
        for row in 0..stride {
            let mut sums = [0.0f32; 4];
            for k in 0..input {
                sums[k % 4] = w[out * input + k].mul_add(x[k * stride + row], sums[k % 4]);
            }
            y[out * stride + row] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2(x: &[f32], w: &[f32], y: &mut [f32], stride: usize, input: usize, output: usize) {
    use std::arch::x86_64::*;
    // Four independent FP32 partial sums limit long-dot rounding error. Reduction order
    // is fixed and shared by the scalar arm, including non-multiple-of-four input tails.
    unsafe {
        let mut out = 0;
        while out + 2 <= output {
            for row in (0..stride).step_by(8) {
                let mut a = [_mm256_setzero_ps(); 4];
                let mut b = [_mm256_setzero_ps(); 4];
                let mut k = 0;
                while k + 4 <= input {
                    for bank in 0..4 {
                        let values = _mm256_loadu_ps(x.as_ptr().add((k + bank) * stride + row));
                        a[bank] = _mm256_fmadd_ps(
                            values,
                            _mm256_set1_ps(w[out * input + k + bank]),
                            a[bank],
                        );
                        b[bank] = _mm256_fmadd_ps(
                            values,
                            _mm256_set1_ps(w[(out + 1) * input + k + bank]),
                            b[bank],
                        );
                    }
                    k += 4;
                }
                for tail in k..input {
                    let values = _mm256_loadu_ps(x.as_ptr().add(tail * stride + row));
                    a[tail % 4] =
                        _mm256_fmadd_ps(values, _mm256_set1_ps(w[out * input + tail]), a[tail % 4]);
                    b[tail % 4] = _mm256_fmadd_ps(
                        values,
                        _mm256_set1_ps(w[(out + 1) * input + tail]),
                        b[tail % 4],
                    );
                }
                let ay = _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
                let by = _mm256_add_ps(_mm256_add_ps(b[0], b[1]), _mm256_add_ps(b[2], b[3]));
                _mm256_storeu_ps(y.as_mut_ptr().add(out * stride + row), ay);
                _mm256_storeu_ps(y.as_mut_ptr().add((out + 1) * stride + row), by);
            }
            out += 2;
        }
        for col in out..output {
            for row in (0..stride).step_by(8) {
                let mut sums = [_mm256_setzero_ps(); 4];
                for k in 0..input {
                    sums[k % 4] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.as_ptr().add(k * stride + row)),
                        _mm256_set1_ps(w[col * input + k]),
                        sums[k % 4],
                    );
                }
                let sum = _mm256_add_ps(
                    _mm256_add_ps(sums[0], sums[1]),
                    _mm256_add_ps(sums[2], sums[3]),
                );
                _mm256_storeu_ps(y.as_mut_ptr().add(col * stride + row), sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(x: &[f32], w: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * output];
        for row in 0..rows {
            for out in 0..output {
                let mut sums = [0.0f32; 4];
                for k in 0..input {
                    sums[k % 4] = w[out * input + k].mul_add(x[row * input + k], sums[k % 4]);
                }
                y[row * output + out] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            }
        }
        y
    }

    #[test]
    fn column_blocking_is_bit_identical_so_thread_count_cannot_move_a_number() {
        let (rows, input, output) = (24usize, 320usize, 11usize);
        let stride = rows.div_ceil(8) * 8;
        let x: Vec<_> = (0..input * stride)
            .map(|i| ((i * 13 % 67) as f32 - 30.0) / 31.0)
            .collect();
        let w: Vec<_> = (0..output * input)
            .map(|i| ((i * 17 % 61) as f32 - 29.0) / 37.0)
            .collect();
        let mut whole = vec![0.0f32; output * stride];
        block(&x, &w, &mut whole, stride, input, output);
        for columns in [2usize, 4, 6, 10] {
            let mut split = vec![0.0f32; output * stride];
            for (wc, yc) in w
                .chunks(columns * input)
                .zip(split.chunks_mut(columns * stride))
            {
                block(&x, wc, yc, stride, input, wc.len() / input);
            }
            for (i, (a, b)) in whole.iter().zip(&split).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "columns {columns} index {i}");
            }
        }
    }

    #[test]
    fn grouping_is_bit_identical_across_group_and_lane_boundaries() {
        // input 2048 gives a 96-row group, so 200 rows cross three groups and the last one
        // is a short group holding the zero padding lanes.
        for &(rows, input, output) in &[(200usize, 2048usize, 5usize), (13, 2048, 3), (96, 2048, 2)]
        {
            assert!(row_group(input, rows.div_ceil(8) * 8) < rows.div_ceil(8) * 8 || rows <= 96);
            let x: Vec<_> = (0..rows * input)
                .map(|i| ((i * 13 % 67) as f32 - 30.0) / 31.0)
                .collect();
            let w: Vec<_> = (0..output * input)
                .map(|i| ((i * 17 % 61) as f32 - 29.0) / 37.0)
                .collect();
            let got = linear(&x, &w, rows, input, output);
            let want = reference(&x, &w, rows, input, output);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "shape {rows}x{input}x{output} index {i}"
                );
            }
        }
    }
    #[test]
    fn independent_row_vectorization_preserves_fma_order_with_tails() {
        let (rows, input, output) = (13, 19, 7);
        let x: Vec<_> = (0..rows * input)
            .map(|i| ((i * 13 % 67) as f32 - 30.0) / 31.0)
            .collect();
        let w: Vec<_> = (0..output * input)
            .map(|i| ((i * 17 % 61) as f32 - 29.0) / 37.0)
            .collect();
        let got = linear(&x, &w, rows, input, output);
        for row in 0..rows {
            for out in 0..output {
                let mut sums = [0.0f32; 4];
                for k in 0..input {
                    sums[k % 4] = w[out * input + k].mul_add(x[row * input + k], sums[k % 4]);
                }
                let expected = (sums[0] + sums[1]) + (sums[2] + sums[3]);
                assert_eq!(got[row * output + out].to_bits(), expected.to_bits());
            }
        }
    }
}
