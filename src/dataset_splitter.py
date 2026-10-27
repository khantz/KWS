import random
import shutil
from pathlib import Path


class DatasetSplitter:
    """Split a dataset into train, validation, and test sets."""

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def run(self) -> None:
        """Run the complete dataset splitting process."""

        random.seed(self.seed)

        class_directories = self._find_class_directories()

        for class_directory in class_directories:
            class_name = class_directory.name

            audio_files = self._find_audio_files(class_directory)

            train_files, val_files, test_files = self._split_audio_files(
                audio_files
            )

            self._copy_files_to_split(
                train_files,
                split_name="train",
                class_name=class_name,
            )

            self._copy_files_to_split(
                val_files,
                split_name="val",
                class_name=class_name,
            )

            self._copy_files_to_split(
                test_files,
                split_name="test",
                class_name=class_name,
            )

            self._print_split_summary(
                class_name,
                train_files,
                val_files,
                test_files,
            )

    def _find_class_directories(self) -> list[Path]:
        """Find all class directories in the source dataset."""

        return [
            directory
            for directory in self.source_dir.iterdir()
            if directory.is_dir()
        ]

    def _find_audio_files(
        self,
        class_directory: Path,
    ) -> list[Path]:
        """Find all WAV files belonging to one class."""

        return list(class_directory.glob("*.wav"))

    def _split_audio_files(
        self,
        audio_files: list[Path],
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Shuffle and divide audio files into three dataset splits."""

        random.shuffle(audio_files)

        total_files = len(audio_files)

        train_count = int(total_files * self.train_ratio)
        val_count = int(total_files * self.val_ratio)

        train_files = audio_files[:train_count]

        val_start = train_count
        val_end = train_count + val_count

        val_files = audio_files[val_start:val_end]
        test_files = audio_files[val_end:]

        return train_files, val_files, test_files

    def _copy_files_to_split(
        self,
        files: list[Path],
        split_name: str,
        class_name: str,
    ) -> None:
        """Copy audio files into a split/class directory."""

        split_directory = (
            self.output_dir
            / split_name
            / class_name
        )

        split_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for file in files:
            destination = split_directory / file.name
            shutil.copy2(file, destination)

    def _print_split_summary(
        self,
        class_name: str,
        train_files: list[Path],
        val_files: list[Path],
        test_files: list[Path],
    ) -> None:
        """Print the number of files assigned to each split."""

        print(
            f"{class_name}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )