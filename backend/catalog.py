import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "documents" / "source_manifest.csv"
RAW_PATH = PROJECT_ROOT / "documents" / "raw"
SUPPORTED_SUFFIXES = {".hwpx", ".pdf", ".txt"}
ASSET_PREFERENCE = {".hwpx": 0, ".txt": 1, ".pdf": 2}


def normalized_title(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).casefold()


@dataclass(frozen=True)
class DocumentAsset:
    source_path: str
    document_type: str
    title: str
    logical_id: str
    version_label: str
    effective_date: str | None
    source_name: str
    source_url: str
    include_in_index: bool
    pdf_path: str | None = None

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.source_path

    def index_metadata(self) -> dict:
        metadata = asdict(self)
        metadata.pop("include_in_index")
        return metadata


def metadata_from_manifest_row(row: dict[str, str]) -> DocumentAsset:
    source_path = row["path"]
    stem = Path(source_path).stem
    title = stem.split("(", 1)[0].strip()
    groups = re.findall(r"\(([^()]*)\)", stem)
    authority = next(
        (
            "고시" if value.endswith("고시") else value
            for value in groups
            if value in {"헌법", "법률", "대통령령", "총리령", "부령", "고시"}
            or value.endswith("고시")
        ),
        "",
    )
    number = next((value for value in groups if re.fullmatch(r"제\d+-?\d*호", value)), "")
    date = next((value for value in reversed(groups) if re.fullmatch(r"\d{8}", value)), None)
    version_label = " ".join(value for value in (authority, number) if value)
    document_type = row["document_type"]
    logical_id = f"{document_type}:{normalized_title(title)}"
    return DocumentAsset(
        source_path=source_path,
        document_type=document_type,
        title=title,
        logical_id=logical_id,
        version_label=version_label,
        effective_date=date,
        source_name=row["source_name"],
        source_url=row["source_url"],
        include_in_index=row["include_in_index"].strip().casefold() == "true",
    )


def load_assets() -> list[DocumentAsset]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as manifest:
        assets = [metadata_from_manifest_row(row) for row in csv.DictReader(manifest)]

    pdf_by_version = {
        (asset.logical_id, asset.version_label, asset.effective_date): asset.source_path
        for asset in assets
        if asset.path.suffix.casefold() == ".pdf" and asset.path.is_file()
    }
    return [
        DocumentAsset(
            **{
                **asdict(asset),
                "pdf_path": pdf_by_version.get(
                    (asset.logical_id, asset.version_label, asset.effective_date)
                ),
            }
        )
        for asset in assets
    ]


def selected_index_assets() -> list[DocumentAsset]:
    active = [
        asset
        for asset in load_assets()
        if asset.include_in_index
        and asset.path.is_file()
        and asset.path.suffix.casefold() in SUPPORTED_SUFFIXES
    ]
    grouped: dict[tuple[str, str, str | None], list[DocumentAsset]] = {}
    for asset in active:
        grouped.setdefault(
            (asset.logical_id, asset.version_label, asset.effective_date), []
        ).append(asset)
    return sorted(
        (
            min(group, key=lambda asset: ASSET_PREFERENCE[asset.path.suffix.casefold()])
            for group in grouped.values()
        ),
        key=lambda asset: asset.source_path,
    )


def unregistered_raw_assets() -> list[str]:
    registered = {asset.source_path for asset in load_assets()}
    return sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in RAW_PATH.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_SUFFIXES
        and path.relative_to(PROJECT_ROOT).as_posix() not in registered
    )


def manifest_errors() -> list[str]:
    errors = []
    for asset in load_assets():
        if not asset.include_in_index:
            continue
        if not asset.path.is_file():
            errors.append(f"active asset is missing: {asset.source_path}")
        elif asset.path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            errors.append(f"active asset has unsupported format: {asset.source_path}")
    errors.extend(
        f"raw asset is not registered: {source_path}"
        for source_path in unregistered_raw_assets()
    )
    return errors
