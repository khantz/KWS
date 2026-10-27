from dataclasses import dataclass, field
from pathlib import Path
import csv

import numpy as np
import soundfile as sf


@dataclass
class AudioValidationResult:
    """Store the validation result of one audio file."""

    file_path: Path
    class_name: str
    is_valid: bool
    status: str

    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    rms: float | None = None
    peak: float | None = None

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class AudioValidator:
    """Validate raw audio files before preprocessing."""

    def __init__(
        self,
        target_sample_rate: int,
        target_channels: int,
        target_duration_seconds: float,
        low_rms_threshold: float,
        clipping_threshold: float,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_duration_seconds = target_duration_seconds
        self.low_rms_threshold = low_rms_threshold
        self.clipping_threshold = clipping_threshold

    def validate(
        self,
        file_path: str | Path,
    ) -> AudioValidationResult:
        """
        Validate one audio file.

        Validation does not modify the original audio.
        """

        file_path = Path(file_path)
        class_name = file_path.parent.name

        try:
            audio, sample_rate = sf.read(
                file_path,
                always_2d=True,
            )

        except Exception as exc:
            return AudioValidationResult(
                file_path=file_path,
                class_name=class_name,
                is_valid=False,
                status="REJECT",
                errors=[
                    f"Unable to read audio: {exc}"
                ],
            )

        channels = audio.shape[1]

        if audio.size == 0:
            return AudioValidationResult(
                file_path=file_path,
                class_name=class_name,
                is_valid=False,
                status="REJECT",
                sample_rate=sample_rate,
                channels=channels,
                errors=[
                    "Audio contains no samples."
                ],
            )

        if not np.isfinite(audio).all():
            return AudioValidationResult(
                file_path=file_path,
                class_name=class_name,
                is_valid=False,
                status="REJECT",
                sample_rate=sample_rate,
                channels=channels,
                errors=[
                    "Audio contains NaN or Inf values."
                ],
            )

        signal = self._to_mono_for_analysis(audio)

        duration_seconds = len(signal) / sample_rate

        rms = float(
            np.sqrt(np.mean(signal ** 2))
        )

        peak = float(
            np.max(np.abs(signal))
        )

        warnings = []
        errors = []

        self._check_sample_rate(
            sample_rate,
            warnings,
        )

        self._check_channels(
            channels,
            warnings,
        )

        self._check_duration(
            duration_seconds,
            warnings,
        )

        self._check_rms(
            rms,
            errors,
            warnings,
        )

        self._check_clipping(
            signal,
            warnings,
        )

        if errors:
            return AudioValidationResult(
                file_path=file_path,
                class_name=class_name,
                is_valid=False,
                status="REJECT",
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration_seconds,
                rms=rms,
                peak=peak,
                warnings=warnings,
                errors=errors,
            )

        status = (
            "WARNING"
            if warnings
            else "VALID"
        )

        return AudioValidationResult(
            file_path=file_path,
            class_name=class_name,
            is_valid=True,
            status=status,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            rms=rms,
            peak=peak,
            warnings=warnings,
            errors=[],
        )

    def validate_directory(
        self,
        directory_path: str | Path,
    ) -> list[AudioValidationResult]:
        """
        Validate all WAV files inside a directory recursively.
        """

        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Directory does not exist: "
                f"{directory_path}"
            )

        if not directory_path.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {directory_path}"
            )

        audio_files = sorted(
            directory_path.rglob("*.wav")
        )

        print(
            f"Found {len(audio_files):,} WAV files."
        )
        print("Starting audio validation...\n")

        results = []

        for index, file_path in enumerate(
            audio_files,
            start=1,
        ):
            result = self.validate(file_path)
            results.append(result)

            if index % 500 == 0:
                print(
                    f"Validated "
                    f"{index:,}/"
                    f"{len(audio_files):,} files..."
                )

        print(
            "\nAudio validation completed."
        )

        return results

    @staticmethod
    def _to_mono_for_analysis(
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        Convert audio to mono only for analysis.

        The original audio is not modified.
        """

        if audio.shape[1] == 1:
            return audio[:, 0]

        return np.mean(audio, axis=1)

    def _check_sample_rate(
        self,
        sample_rate: int,
        warnings: list[str],
    ) -> None:
        if sample_rate != self.target_sample_rate:
            warnings.append(
                f"Sample rate is {sample_rate} Hz; "
                f"target is "
                f"{self.target_sample_rate} Hz."
            )

    def _check_channels(
        self,
        channels: int,
        warnings: list[str],
    ) -> None:
        if channels != self.target_channels:
            warnings.append(
                f"Audio has {channels} channels; "
                f"target is "
                f"{self.target_channels} channel(s)."
            )

    def _check_duration(
        self,
        duration_seconds: float,
        warnings: list[str],
    ) -> None:
        if not np.isclose(
            duration_seconds,
            self.target_duration_seconds,
            atol=1e-6,
        ):
            warnings.append(
                f"Duration is "
                f"{duration_seconds:.3f} s; "
                f"target is "
                f"{self.target_duration_seconds:.1f} s."
            )

    def _check_rms(
        self,
        rms: float,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if rms == 0.0:
            errors.append(
                "Audio is completely silent."
            )
            return

        if rms < self.low_rms_threshold:
            warnings.append(
                f"Very low RMS level: "
                f"{rms:.6f}."
            )

    def _check_clipping(
        self,
        signal: np.ndarray,
        warnings: list[str],
    ) -> None:
        clipped_samples = int(
            np.sum(
                np.abs(signal)
                >= self.clipping_threshold
            )
        )

        if clipped_samples > 0:
            warnings.append(
                "Potential clipping detected: "
                f"{clipped_samples} samples at or "
                f"above "
                f"{self.clipping_threshold:.3f}."
            )

    @staticmethod
    def summarize_results(
        results: list[AudioValidationResult],
    ) -> dict[str, int]:
        """Return validation counts."""

        summary = {
            "VALID": 0,
            "WARNING": 0,
            "REJECT": 0,
        }

        for result in results:
            summary[result.status] += 1

        return summary

    @staticmethod
    def print_summary(
        results: list[AudioValidationResult],
    ) -> None:
        """Print validation summary."""

        summary = AudioValidator.summarize_results(
            results
        )

        print("\n" + "=" * 40)
        print("Audio Validation Summary")
        print("=" * 40)

        print(
            f"Total files : {len(results):,}"
        )
        print(
            f"VALID       : {summary['VALID']:,}"
        )
        print(
            f"WARNING     : {summary['WARNING']:,}"
        )
        print(
            f"REJECT      : {summary['REJECT']:,}"
        )

        print("=" * 40)

    @staticmethod
    def print_issues(
        results: list[AudioValidationResult],
    ) -> None:
        """Print warning and rejected files."""

        warning_results = [
            result
            for result in results
            if result.status == "WARNING"
        ]

        rejected_results = [
            result
            for result in results
            if result.status == "REJECT"
        ]

        print("\n" + "=" * 40)
        print("WARNING FILES")
        print("=" * 40)

        if warning_results:
            for result in warning_results:
                print(
                    f"\n[{result.class_name}] "
                    f"{result.file_path.name}"
                )

                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print("No warning files.")

        print("\n" + "=" * 40)
        print("REJECTED FILES")
        print("=" * 40)

        if rejected_results:
            for result in rejected_results:
                print(
                    f"\n[{result.class_name}] "
                    f"{result.file_path.name}"
                )

                for error in result.errors:
                    print(f"  - {error}")

                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print("No rejected files.")

    @staticmethod
    def save_report(
        results: list[AudioValidationResult],
        output_path: str | Path,
    ) -> None:
        """Save validation results as a CSV file."""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "class_name",
            "filename",
            "relative_path",
            "status",
            "sample_rate",
            "channels",
            "duration_seconds",
            "rms",
            "peak",
            "warnings",
            "errors",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:
                writer.writerow(
                    {
                        "class_name": result.class_name,
                        "filename": result.file_path.name,
                        "relative_path": (
                            result.file_path.as_posix()
                        ),
                        "status": result.status,
                        "sample_rate": (
                            result.sample_rate
                            if result.sample_rate is not None
                            else ""
                        ),
                        "channels": (
                            result.channels
                            if result.channels is not None
                            else ""
                        ),
                        "duration_seconds": (
                            f"{result.duration_seconds:.4f}"
                            if result.duration_seconds
                            is not None
                            else ""
                        ),
                        "rms": (
                            f"{result.rms:.6f}"
                            if result.rms is not None
                            else ""
                        ),
                        "peak": (
                            f"{result.peak:.6f}"
                            if result.peak is not None
                            else ""
                        ),
                        "warnings": " | ".join(
                            result.warnings
                        ),
                        "errors": " | ".join(
                            result.errors
                        ),
                    }
                )

        print(
            "\nValidation report saved to:"
        )
        print(output_path)