"""
data_loader.py
--------------
Loads and parses Microsoft MIND-small files into pandas DataFrames.

MIND files (TSV, no headers):
  behaviors.tsv: impression_id, user_id, time, history, impressions
  news.tsv:      news_id, category, subcategory, title, abstract,
                 url, title_entities, abstract_entities

The MIND-small zip also contains entity_embedding.vec and relation_embedding.vec.
This project does NOT use them — we build our own topic features from titles
and abstracts via TF-IDF + LDA.

Public functions:
  setup_mind_data(...)               -> path to extracted MIND folder (Drive-aware)
  load_mind(data_dir)                -> (behaviors_df, news_df)
  parse_impressions(impressions_str) -> list[(news_id, label)]
  explode_impressions(behaviors_df)  -> long-format candidate rows
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Defining diffeerent dataset columns
BEHAVIORS_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]


# Default Drive location used by the project. Override via the function arg.
# Link to Google Drive (public) - https://drive.google.com/drive/folders/1qOv2KSUd_vroeu8hZTXgMpFpzs76P1nY
# Data has been uploaded to this location for easy access in Colab 
# and to avoid any firewall issues by accessing directly from MIND website.
DEFAULT_DRIVE_ZIP = (
    "/content/drive/MyDrive/Data Mining/Project/Datasets/MINDsmall_train.zip"
)
DEFAULT_EXTRACT_DIR = "MINDsmall_train"


def setup_mind_data(
    drive_zip_path: str = DEFAULT_DRIVE_ZIP,
    extract_dir: str = DEFAULT_EXTRACT_DIR,
    local_zip_path: Optional[str] = None,
) -> str:
    """Locate or unpack MIND-small data and return the folder path.

    The function tries three sources, in order:

      1. If ``extract_dir`` already contains behaviors.tsv + news.tsv, use it.
         (Re-runs are free — no re-extraction.)
      2. If ``local_zip_path`` is provided and exists, extract from there.
         (Useful for local development outside Colab.)
      3. If running in Colab, mount Google Drive and extract from
         ``drive_zip_path``. This mirrors the Checkpoint 1 setup cell.

    Parameters
    ----------
    drive_zip_path : str
        Path to MINDsmall_train.zip inside the user's Google Drive.
        Defaults to the project's standard location.
    extract_dir : str
        Folder name to extract the zip into. Defaults to "MINDsmall_train".
    local_zip_path : str, optional
        Path to a local copy of MINDsmall_train.zip (e.g. data/MINDsmall_train.zip).

    Returns
    -------
    str
        Path to the folder containing behaviors.tsv and news.tsv.
        Pass this directly to :func:`load_mind`.
    """
    # 1) Already extracted?
    if _has_required_files(extract_dir):
        print(f"MIND data already extracted at: {extract_dir}")
        return extract_dir

    # 2) Local zip?
    if local_zip_path and os.path.exists(local_zip_path):
        print(f"Extracting from local zip: {local_zip_path}")
        _extract_zip(local_zip_path, extract_dir)
        return extract_dir

    # 3) Try Google Drive (Colab)
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        print("Google Drive mounted.")
    except ImportError:
        raise FileNotFoundError(
            f"MIND data not found at {extract_dir!r}. "
            "Either: (a) run in Colab with the zip at "
            f"{drive_zip_path!r}, (b) pass local_zip_path=..., "
            "or (c) manually unzip MINDsmall_train.zip into "
            f"{extract_dir!r}."
        )

    if not os.path.exists(drive_zip_path):
        raise FileNotFoundError(
            f"MINDsmall_train.zip not found at {drive_zip_path!r}. "
            "Upload it to that Drive location, or pass a different "
            "drive_zip_path."
        )

    print(f"Extracting from Drive: {drive_zip_path}")
    _extract_zip(drive_zip_path, extract_dir)
    return extract_dir


def _extract_zip(zip_path: str, extract_dir: str) -> None:
    """
        Validate and extract a MIND-small zip; verify required files appear.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path!r} is not a valid ZIP archive.")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Detect leading subfolder prefix by finding behaviors.tsv in the archive
        prefix = ""
        for n in names:
            if n.endswith("behaviors.tsv"):
                prefix = n[: -len("behaviors.tsv")]  # e.g. "MINDsmall_train/"
                break

        if prefix:
            # Layout B: strip leading subfolder, extract flat into extract_dir
            for member in zf.infolist():
                if member.filename == prefix:
                    continue
                relative = member.filename[len(prefix):]
                if not relative:
                    continue
                target = os.path.join(extract_dir, relative)
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        else:
            # Layout A: files already at archive root
            zf.extractall(extract_dir)

    if not _has_required_files(extract_dir):
        raise FileNotFoundError(
            f"Extraction completed but behaviors.tsv / news.tsv are missing "
            f"in {extract_dir!r}. Zip contents: {names[:10]}"
        )
    print(f"Extraction complete. Files available in: {extract_dir}")


def _has_required_files(folder: str) -> bool:
    return (
        os.path.exists(os.path.join(folder, "behaviors.tsv"))
        and os.path.exists(os.path.join(folder, "news.tsv"))
    )


def load_mind(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load behaviors.tsv and news.tsv from a MIND-small folder.

    Parameters
    ----------
    data_dir : path-like
        Folder containing behaviors.tsv and news.tsv.

    Returns
    -------
    behaviors : pd.DataFrame
        Columns: impression_id, user_id, time (datetime), history (str),
                 impressions (str), history_len (int).
    news : pd.DataFrame
        News metadata columns + a combined `text` column (title + abstract).
    """
    data_dir = Path(data_dir)
    behaviors_path = data_dir / "behaviors.tsv"
    news_path = data_dir / "news.tsv"

    if not behaviors_path.exists():
        raise FileNotFoundError(
            f"behaviors.tsv not found at {behaviors_path}. "
            "Use setup_mind_data() first to mount Drive / extract the zip, "
            "or unzip MINDsmall_train.zip manually."
        )
    if not news_path.exists():
        raise FileNotFoundError(f"news.tsv not found at {news_path}.")

    behaviors = pd.read_csv(
        behaviors_path, sep="\t", header=None, names=BEHAVIORS_COLS,
    )
    news = pd.read_csv(
        news_path, sep="\t", header=None, names=NEWS_COLS,
    )

    # Parse time. MIND uses "M/D/YYYY h:mm:ss AM/PM"; fall back to flexible
    # parsing if the format ever changes.
    parsed = pd.to_datetime(
        behaviors["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    if parsed.isna().any():
        parsed = pd.to_datetime(behaviors["time"], errors="coerce")
    behaviors["time"] = parsed

    # Computing history length (zero = cold-start)
    behaviors["history"] = behaviors["history"].fillna("")
    behaviors["history_len"] = behaviors["history"].str.split().str.len().fillna(0).astype(int)

    # Combined text field on news (title + abstract); used for TF-IDF / LDA
    news["abstract"] = news["abstract"].fillna("")
    news["title"] = news["title"].fillna("")
    news["text"] = (news["title"] + " " + news["abstract"]).str.strip()

    return behaviors, news


def parse_impressions(impressions_str: str) -> List[Tuple[str, int]]:
    """
    Parse 'N1-0 N2-1 N3-0' into [(N1, 0), (N2, 1), (N3, 0)].

    Robust to NaN/empty strings.
    """
    if not isinstance(impressions_str, str) or not impressions_str.strip():
        return []
    out = []
    for token in impressions_str.split():
        if "-" not in token:
            continue
        nid, label = token.rsplit("-", 1)
        try:
            out.append((nid, int(label)))
        except ValueError:
            continue
    return out


def explode_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """
    Convert behaviors (one row per impression) to long format
    (one row per candidate article in each impression).

    Returns a DataFrame with:
      impression_id, user_id, time, history, history_len,
      candidate_id, label, position
    """
    rows = []
    for rec in behaviors[
        ["impression_id", "user_id", "time", "history", "history_len", "impressions"]
    ].itertuples(index=False):
        for pos, (nid, label) in enumerate(parse_impressions(rec.impressions)):
            rows.append({
                "impression_id": rec.impression_id,
                "user_id": rec.user_id,
                "time": rec.time,
                "history": rec.history,
                "history_len": rec.history_len,
                "candidate_id": nid,
                "label": label,
                "position": pos,
            })
    return pd.DataFrame(rows)