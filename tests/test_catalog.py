from backend.catalog import manifest_errors, selected_index_assets, unregistered_raw_assets


def test_manifest_registers_every_supported_raw_asset():
    assert unregistered_raw_assets() == []
    assert manifest_errors() == []


def test_different_law_versions_remain_distinct_assets():
    criminal_procedure = [
        asset for asset in selected_index_assets()
        if asset.title == "형사소송법"
    ]

    assert {asset.version_label for asset in criminal_procedure} == {
        "법률 제20796호",
        "법률 제21232호",
    }
    older = next(asset for asset in criminal_procedure if "제20796호" in asset.version_label)
    assert older.pdf_path is None
