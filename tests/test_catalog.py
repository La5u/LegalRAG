import pytest

from backend.catalog import manifest_errors, selected_index_assets, unregistered_raw_assets


def test_manifest_registers_every_supported_raw_asset():
    assert unregistered_raw_assets() == []
    assert manifest_errors() == []


def test_only_current_criminal_procedure_version_is_selected():
    criminal_procedure = [
        asset for asset in selected_index_assets()
        if asset.title == "형사소송법"
    ]

    if not criminal_procedure:
        pytest.skip("형사소송법 corpus files are not installed locally")

    assert [asset.version_label for asset in criminal_procedure] == ["법률 제21232호"]


def test_missing_assets_are_reported_without_manifest_errors(monkeypatch, tmp_path):
    from backend import catalog

    asset = catalog.metadata_from_manifest_row({
        "path": str(tmp_path / "missing.txt"), "document_type": "contract",
        "source_name": "test", "source_url": "", "include_in_index": "true",
    })
    monkeypatch.setattr(catalog, "load_assets", lambda: [asset])
    monkeypatch.setattr(catalog, "unregistered_raw_assets", lambda: [])
    assert catalog.missing_assets() == [asset]
    assert catalog.manifest_errors() == []
    assert catalog.selected_index_assets() == []


def test_unsupported_missing_asset_is_still_an_error(monkeypatch, tmp_path):
    from backend import catalog

    asset = catalog.metadata_from_manifest_row({
        "path": str(tmp_path / "missing.doc"), "document_type": "contract",
        "source_name": "test", "source_url": "", "include_in_index": "true",
    })
    monkeypatch.setattr(catalog, "load_assets", lambda: [asset])
    monkeypatch.setattr(catalog, "unregistered_raw_assets", lambda: [])
    assert "unsupported format" in catalog.manifest_errors()[0]
