"""Compare FFTW3f and oneMKL real FFT with the pinned Torch audio corpus.

This is a diagnostic, not a Memra runtime dependency. Pass the paths to
libfftw3f and either standalone libmkl_rt.so.2 or the Torch CPU wheel's
libtorch_cpu.so. Both export the oneMKL DFTI functions. Requires
torch/torchaudio 2.6.0 CPU to form and compare the publisher mel output.
"""

import ctypes
from pathlib import Path
import struct
import sys

import torch
import torchaudio


N_FFT = 960
N_FREQ = N_FFT // 2 + 1
FFTW_ESTIMATE = 1 << 6


class FftwRealFft:
    def __init__(self, library_path: str):
        lib = ctypes.CDLL(library_path)
        lib.fftwf_plan_dft_r2c_1d.argtypes = (
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint
        )
        lib.fftwf_plan_dft_r2c_1d.restype = ctypes.c_void_p
        lib.fftwf_execute.argtypes = (ctypes.c_void_p,)
        lib.fftwf_destroy_plan.argtypes = (ctypes.c_void_p,)
        self.lib = lib
        self.input = (ctypes.c_float * N_FFT)()
        self.output = (ctypes.c_float * (2 * N_FREQ))()
        self.plan = lib.fftwf_plan_dft_r2c_1d(
            N_FFT, ctypes.addressof(self.input),
            ctypes.addressof(self.output), FFTW_ESTIMATE,
        )
        if not self.plan:
            raise RuntimeError("FFTW3f could not plan the 960-point real FFT")

    def transform(self, frame: torch.Tensor) -> torch.Tensor:
        for index, sample in enumerate(frame.tolist()):
            self.input[index] = sample
        self.lib.fftwf_execute(self.plan)
        real = torch.tensor([self.output[2 * index] for index in range(N_FREQ)])
        imag = torch.tensor([self.output[2 * index + 1] for index in range(N_FREQ)])
        return torch.complex(real, imag)

    def close(self):
        self.lib.fftwf_destroy_plan(self.plan)


class MklRealFft:
    def __init__(self, library_path: str):
        lib = ctypes.CDLL(library_path)
        lib.DftiCreateDescriptor.argtypes = (
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
            ctypes.c_int, ctypes.c_longlong,
        )
        lib.DftiCreateDescriptor.restype = ctypes.c_longlong
        lib.DftiSetValue.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.DftiSetValue.restype = ctypes.c_longlong
        lib.DftiCommitDescriptor.argtypes = (ctypes.c_void_p,)
        lib.DftiCommitDescriptor.restype = ctypes.c_longlong
        lib.DftiComputeForward.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )
        lib.DftiComputeForward.restype = ctypes.c_longlong
        lib.DftiFreeDescriptor.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
        lib.DftiFreeDescriptor.restype = ctypes.c_longlong
        self.lib = lib
        self.handle = ctypes.c_void_p()
        self.check(lib.DftiCreateDescriptor(
            ctypes.byref(self.handle), 35, 33, 1, ctypes.c_longlong(N_FFT),
        ))
        self.check(lib.DftiSetValue(self.handle, 11, ctypes.c_int(44)))
        self.check(lib.DftiSetValue(self.handle, 10, ctypes.c_int(39)))
        self.check(lib.DftiCommitDescriptor(self.handle))
        self.input = (ctypes.c_float * N_FFT)()
        self.output = (ctypes.c_float * (2 * N_FREQ))()

    @staticmethod
    def check(status: int):
        if status:
            raise RuntimeError(f"oneMKL DFTI status {status}")

    def transform(self, frame: torch.Tensor) -> torch.Tensor:
        for index, sample in enumerate(frame.tolist()):
            self.input[index] = sample
        self.check(self.lib.DftiComputeForward(
            self.handle, ctypes.addressof(self.input),
            ctypes.addressof(self.output),
        ))
        real = torch.tensor([self.output[2 * index] for index in range(N_FREQ)])
        imag = torch.tensor([self.output[2 * index + 1] for index in range(N_FREQ)])
        return torch.complex(real, imag)

    def close(self):
        self.check(self.lib.DftiFreeDescriptor(ctypes.byref(self.handle)))


def read_f32(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    return torch.tensor(struct.unpack(f"<{len(data) // 4}f", data))


def errors(actual: torch.Tensor, expected: torch.Tensor):
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-6)
    return difference.max().item(), relative.max().item()


def main():
    assert torch.__version__.startswith("2.6.0"), torch.__version__
    assert torchaudio.__version__.startswith("2.6.0"), torchaudio.__version__
    torch.set_num_threads(1)
    fixture_dir = Path(__file__).resolve().parent
    fftw = FftwRealFft(sys.argv[1])
    mkl = MklRealFft(sys.argv[2])
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24_000, n_fft=960, hop_length=240, win_length=960,
        f_min=0, f_max=None, n_mels=128, power=1.0, center=True,
    )
    filters = transform.mel_scale.fb
    window = transform.spectrogram.window
    print(f"torch={torch.__version__} torchaudio={torchaudio.__version__}")
    print("case torch_manual_abs fftw_log_abs fftw_log_rel mkl_log_abs mkl_log_rel mkl_complex_equal")
    try:
        for pcm_path in sorted(fixture_dir.glob("mimo-pcm-mel-*.pcm.f32")):
            name = pcm_path.name.removeprefix("mimo-pcm-mel-").removesuffix(".pcm.f32")
            pcm = read_f32(pcm_path)
            target = read_f32(
                fixture_dir / f"mimo-pcm-mel-{name}.logmel.f32"
            ).reshape(128, -1)
            publisher_from_transform = torch.log(
                transform(pcm[None, :]).clamp_min(1e-7)
            )[0]
            assert torch.equal(publisher_from_transform, target), name
            padded = torch.nn.functional.pad(
                pcm[None, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect"
            )[0, 0]
            windowed = padded.unfold(0, N_FFT, 240) * window
            torch_magnitude = torch.fft.rfft(windowed, dim=-1).abs()
            torch_log = torch.log((torch_magnitude @ filters).clamp_min(1e-7)).T
            manual_abs, _ = errors(torch_log, target)
            fftw_spectrum = torch.stack(
                [fftw.transform(frame) for frame in windowed]
            )
            fftw_log = torch.log((fftw_spectrum.abs() @ filters).clamp_min(1e-7)).T
            fftw_abs, fftw_rel = errors(fftw_log, target)
            mkl_spectrum = torch.stack(
                [mkl.transform(frame) for frame in windowed]
            )
            mkl_log = torch.log((mkl_spectrum.abs() @ filters).clamp_min(1e-7)).T
            mkl_abs, mkl_rel = errors(mkl_log, target)
            mkl_complex_equal = torch.equal(
                mkl_spectrum, torch.fft.rfft(windowed, dim=-1)
            )
            print(
                f"{name} {manual_abs:.9g} "
                f"{fftw_abs:.9g} {fftw_rel:.9g} "
                f"{mkl_abs:.9g} {mkl_rel:.9g} {mkl_complex_equal}"
            )
    finally:
        fftw.close()
        mkl.close()


if __name__ == "__main__":
    main()
