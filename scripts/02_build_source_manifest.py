from pathlib import Path
from datetime import date
import csv
import hashlib

import pandas as pd
import soundfile as sf
from mutagen import File as MutagenFile
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw_sources"
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "candidate_sources"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

SOURCE_MANIFEST_PATH = MANIFEST_DIR / "source_manifest.csv"
SOURCE_SUMMARY_PATH = MANIFEST_DIR / "source_summary.csv"
CANDIDATE_MANIFEST_PATH = MANIFEST_DIR / "candidate_manifest_unclear_license.csv"
REJECTED_SOURCES_PATH = MANIFEST_DIR / "rejected_sources.csv"
DEFERRED_SOURCES_PATH = MANIFEST_DIR / "deferred_large_sources.csv"
ERROR_PATH = MANIFEST_DIR / "manifest_errors.csv"


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aiff",
    ".aif",
}


SOURCE_CONFIG = {
    "mendeley_poultry_vocalization": {
        "dataset_name": "Poultry Vocalization Signal Dataset for Early Disease Detection",
        "source_url": "https://data.mendeley.com/datasets/zp4nf2dxbh/1",
        "license": "CC BY 4.0",
        "license_status": "clear",
        "expected_sample_rate_hz": "96000",
        "include": "yes",
        "notes": "Mendeley public poultry vocalization dataset. Folders: Healthy, Noise, Unhealthy.",
    },
    "zenodo_laying_hens_stress": {
        "dataset_name": "Vocalization Patterns in Laying Hens - An Analysis of Stress-Induced Audio Responses",
        "source_url": "https://zenodo.org/records/10433023",
        "license": "CC BY 4.0",
        "license_status": "clear",
        "expected_sample_rate_hz": "",
        "include": "yes",
        "notes": "Zenodo laying hen control and treatment vocalization data.",
    },
}


CANDIDATE_CONFIG = {
    "ChickenLanguageDataset": {
        "dataset_name": "ChickenLanguageDataset",
        "source_url": "https://github.com/zebular13/ChickenLanguageDataset",
        "license": "unknown",
        "license_status": "unclear",
        "expected_sample_rate_hz": "",
        "include": "no",
        "notes": "Candidate chicken vocalization dataset. Excluded from formal manifest because license is not confirmed.",
    },
    "chicken_language_dataset_unclear_license": {
        "dataset_name": "ChickenLanguageDataset",
        "source_url": "https://github.com/zebular13/ChickenLanguageDataset",
        "license": "unknown",
        "license_status": "unclear",
        "expected_sample_rate_hz": "",
        "include": "no",
        "notes": "Candidate chicken vocalization dataset. Excluded from formal manifest because license is not confirmed.",
    },
}


REJECTED_SOURCES = [
    {
        "dataset_name": "ChickenLanguageDataset",
        "source_url": "https://github.com/zebular13/ChickenLanguageDataset",
        "license": "unknown",
        "license_status": "unclear",
        "decision": "excluded_from_formal_manifest",
        "reason": "License was not confirmed. Files are kept only as candidate data and excluded from source_manifest.csv.",
    }
]


DEFERRED_SOURCES = [
    {
        "dataset_name": "ChickenSense",
        "source_url": "https://zenodo.org/records/8212853",
        "license": "CC BY 4.0",
        "license_status": "clear",
        "decision": "deferred_not_downloaded",
        "reason": "Full Dataset.zip is about 68.9 GB. It is suitable for later large-scale real-world validation, but not required for Day2 manifest construction.",
    }
]


def collect_audio_files(root: Path):
    files = []
    if not root.exists():
        return files

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(file_path)

    return sorted(set(files))


def md5_for_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def infer_source_id(file_path: Path, root: Path) -> str:
    rel = file_path.relative_to(root)
    return rel.parts[0]


def infer_label(file_path: Path) -> str:
    text = " ".join(part.lower() for part in file_path.parts)

    if "unhealthy" in text or "sick" in text or "disease" in text:
        return "unhealthy"

    if "healthy" in text:
        return "healthy"

    if "noise" in text or "noisy" in text:
        return "noise"

    if "control" in text or "ctrl" in text:
        return "control"

    if "treatment 1" in text or "treatment_1" in text or "treatment1" in text:
        return "treatment_1"

    if "treatment 2" in text or "treatment_2" in text or "treatment2" in text:
        return "treatment_2"

    if "treatment" in text or "trs" in text:
        return "treatment"

    if "longer_segments" in text:
        return "longer_segments"

    if "single_vocalizations" in text:
        parts = [p.lower() for p in file_path.parts]
        if "single_vocalizations" in parts:
            idx = parts.index("single_vocalizations")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "single_vocalizations"

    return file_path.parent.name


def read_audio_info(file_path: Path):
    """
    First try soundfile.
    If soundfile cannot read MP3/M4A on this machine, fall back to mutagen.
    """
    try:
        info = sf.info(str(file_path))
        sample_rate = int(info.samplerate)
        frames = int(info.frames)
        channels = int(info.channels)
        duration_sec = frames / sample_rate if sample_rate > 0 else 0.0
        subtype = str(info.subtype)
        backend = "soundfile"
        return sample_rate, duration_sec, channels, subtype, backend
    except Exception as sf_exc:
        audio = MutagenFile(str(file_path))

        if audio is None or audio.info is None:
            raise RuntimeError(f"Cannot read audio info with soundfile or mutagen. soundfile error: {repr(sf_exc)}")

        duration_sec = float(getattr(audio.info, "length", 0.0))
        sample_rate = int(getattr(audio.info, "sample_rate", 0))
        channels = int(getattr(audio.info, "channels", 0))
        subtype = file_path.suffix.lower().replace(".", "")
        backend = "mutagen"

        if duration_sec <= 0:
            raise RuntimeError(f"Invalid duration read by mutagen. soundfile error: {repr(sf_exc)}")

        return sample_rate, duration_sec, channels, subtype, backend


def build_rows(audio_files, root: Path, source_config: dict, is_candidate: bool):
    rows = []
    errors = []

    for file_path in tqdm(audio_files, desc=f"Scanning {root.name}"):
        try:
            source_id = infer_source_id(file_path, root)
            info = source_config.get(source_id)

            if info is None:
                errors.append(
                    {
                        "raw_path": str(file_path.relative_to(PROJECT_ROOT)),
                        "reason": f"source_id_not_registered: {source_id}",
                    }
                )
                continue

            sample_rate, duration_sec, channels, subtype, backend = read_audio_info(file_path)

            include = info["include"]
            exclusion_reason = ""

            if info["license_status"] != "clear":
                include = "no"
                exclusion_reason = "license_unclear"

            rows.append(
                {
                    "source_id": source_id,
                    "dataset_name": info["dataset_name"],
                    "source_url": info["source_url"],
                    "license": info["license"],
                    "license_status": info["license_status"],
                    "download_date": str(date.today()),
                    "raw_path": str(file_path.relative_to(PROJECT_ROOT)),
                    "file_name": file_path.name,
                    "file_extension": file_path.suffix.lower(),
                    "file_md5": md5_for_file(file_path),
                    "original_sample_rate_hz": sample_rate,
                    "channels": channels,
                    "duration_sec": round(duration_sec, 6),
                    "audio_subtype": subtype,
                    "metadata_backend": backend,
                    "label": infer_label(file_path),
                    "include": include,
                    "exclusion_reason": exclusion_reason,
                    "notes": info["notes"],
                }
            )

        except Exception as exc:
            errors.append(
                {
                    "raw_path": str(file_path.relative_to(PROJECT_ROOT)) if file_path.exists() else str(file_path),
                    "reason": f"unexpected_error: {repr(exc)}",
                }
            )

    return rows, errors


def save_static_records(path: Path, records: list):
    if not records:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    official_files = collect_audio_files(RAW_ROOT)
    candidate_files = collect_audio_files(CANDIDATE_ROOT)

    official_rows, official_errors = build_rows(
        official_files,
        RAW_ROOT,
        SOURCE_CONFIG,
        is_candidate=False,
    )

    candidate_rows, candidate_errors = build_rows(
        candidate_files,
        CANDIDATE_ROOT,
        CANDIDATE_CONFIG,
        is_candidate=True,
    )

    all_errors = official_errors + candidate_errors

    official_df = pd.DataFrame(official_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    if len(official_df) > 0:
        official_df = official_df[official_df["include"] == "yes"].copy()
        official_df = official_df.sort_values(["source_id", "label", "raw_path"])
        official_df.to_csv(SOURCE_MANIFEST_PATH, index=False, encoding="utf-8-sig")

        summary = (
            official_df.groupby(["source_id", "dataset_name", "license", "label"])
            .agg(
                file_count=("raw_path", "count"),
                total_duration_sec=("duration_sec", "sum"),
                min_sample_rate_hz=("original_sample_rate_hz", "min"),
                max_sample_rate_hz=("original_sample_rate_hz", "max"),
            )
            .reset_index()
        )
        summary["total_duration_sec"] = summary["total_duration_sec"].round(3)
        summary["total_duration_min"] = (summary["total_duration_sec"] / 60.0).round(3)
        summary.to_csv(SOURCE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    else:
        print("No official audio files were accepted.")

    if len(candidate_df) > 0:
        candidate_df = candidate_df.sort_values(["source_id", "label", "raw_path"])
        candidate_df.to_csv(CANDIDATE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(CANDIDATE_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    save_static_records(REJECTED_SOURCES_PATH, REJECTED_SOURCES)
    save_static_records(DEFERRED_SOURCES_PATH, DEFERRED_SOURCES)

    if all_errors:
        pd.DataFrame(all_errors).to_csv(ERROR_PATH, index=False, encoding="utf-8-sig")
    elif ERROR_PATH.exists():
        ERROR_PATH.unlink()

    print("Manifest build finished.")
    print(f"Official accepted files: {len(official_df)}")
    print(f"Candidate files scanned: {len(candidate_df)}")
    print(f"Saved official manifest: {SOURCE_MANIFEST_PATH}")
    print(f"Saved official summary: {SOURCE_SUMMARY_PATH}")
    print(f"Saved candidate manifest: {CANDIDATE_MANIFEST_PATH}")
    print(f"Saved rejected sources: {REJECTED_SOURCES_PATH}")
    print(f"Saved deferred large sources: {DEFERRED_SOURCES_PATH}")

    if all_errors:
        print(f"Some files had errors. See: {ERROR_PATH}")

    if len(official_df) > 0:
        print()
        print("Official source summary:")
        print(official_df.groupby(["source_id", "label"]).size())
        print()
        print("Official total files:", len(official_df))
        print("Official total duration seconds:", round(official_df["duration_sec"].sum(), 3))
        print("Official source count:", official_df["source_id"].nunique())
        print("Licenses:", official_df["license"].unique())

    if len(candidate_df) > 0:
        print()
        print("Candidate source summary:")
        print(candidate_df.groupby(["source_id", "label"]).size())
        print()
        print("Candidate total files:", len(candidate_df))
        print("Candidate total duration seconds:", round(candidate_df["duration_sec"].sum(), 3))


if __name__ == "__main__":
    main()