from pathlib import Path
from statistics import mean, stdev

import librosa.feature
from numpy.typing import NDArray
from pesq import pesq
from tqdm.notebook import tqdm


def reconstruct_waveform(
        waveform: NDArray,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        window: str,
        power: float,
) -> NDArray:
    return librosa.griffinlim(
        librosa.feature.inverse.mel_to_stft(
            librosa.feature.melspectrogram(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                window=window,
                power=power
            ),
            sr=sample_rate,
            n_fft=n_fft,
            power=power
        ),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )


def run_testing_pipeline(directory: Path, reconstruct: bool = True, sample_rate: int = 16000, **kwargs) -> None:
    score = []
    for filepath in tqdm(directory.iterdir()):

        if not filepath.suffix == ".wav":
            continue

        original_waveform, _ = librosa.load(filepath, sr=sample_rate)

        reconstructed_waveform = reconstruct_waveform(
            original_waveform, sample_rate, **kwargs
        ) if reconstruct else original_waveform

        min_length = min(len(original_waveform), len(reconstructed_waveform))

        score.append(
            pesq(
                fs=sample_rate,
                ref=original_waveform[:min_length],
                deg=reconstructed_waveform[:min_length],
                mode="wb",
            )
        )

    print(f"MEAN PESQ: {mean(score) if len(score) > 1 else None}")
    print(f"STD PESQ: {stdev(score) if len(score) > 2 else None}")
    print(f"MAX PESQ: {max(score)}")
    print(f"MIN PESQ: {min(score)}")
