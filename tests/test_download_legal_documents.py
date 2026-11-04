import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_legal_documents.py"
spec = importlib.util.spec_from_file_location("download_legal_documents", SCRIPT)
download_legal_documents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_legal_documents)


def test_html_to_paragraphs_drops_markup_inside_quoted_attributes():
    html = (
        "<p>제1조(목적) 이 기준은 &lsquo;법&rsquo;에 따른다.</p>"
        "<a onclick=\"show('<img src=&quot;/LSW/x.gif&quot; />');return false;\" "
        "href=\"#AJAX\"><img alt=\"조문목록 없음\" src=\"/LSW/x.gif\" /></a>"
        "<div>① 첫째 항<br>② 둘째 항</div>"
    )

    assert download_legal_documents.html_to_paragraphs(html) == (
        "제1조(목적) 이 기준은 ‘법’에 따른다.\n\n① 첫째 항\n\n② 둘째 항\n"
    )
