//! Single-threaded native matrix product for the speech reference executor.
//! Fixed four-bank FP32 FMA order; AVX2 vectorizes independent rows, not the reduction.

pub(super) fn linear(x: &[f32], w: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * input);
    assert_eq!(w.len(), output * input);
    let stride = rows.div_ceil(8) * 8;
    let mut xt = vec![0.0f32; input * stride];
    for row in 0..rows {
        for k in 0..input {
            xt[k * stride + row] = x[row * input + k];
        }
    }
    let mut yt = vec![0.0f32; output * stride];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: features checked above; xt/yt are padded to eight complete row lanes.
        unsafe {
            avx2(&xt, w, &mut yt, stride, input, output);
        }
    } else {
        scalar(&xt, w, &mut yt, stride, input, output);
    }
    #[cfg(not(target_arch = "x86_64"))]
    scalar(&xt, w, &mut yt, stride, input, output);
    let mut y = vec![0.0f32; rows * output];
    for row in 0..rows {
        for col in 0..output {
            y[row * output + col] = yt[col * stride + row];
        }
    }
    y
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
