"""Download the reviewed legal-corpus plan from approved official sites."""

import json
import re
import sys
from datetime import date
from html import unescape
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = PROJECT_ROOT / "documents" / "download_plan.json"
DOCUMENTS_PATH = (PROJECT_ROOT / "documents").resolve()
ALLOWED_HOSTS = {
    "law.go.kr",
    "www.law.go.kr",
    "help.scourt.go.kr",
    "www.scourt.go.kr",
    "ecfs.scourt.go.kr",
    "moel.go.kr",
    "www.moel.go.kr",
    "mss.go.kr",
    "www.mss.go.kr",
    "ftc.go.kr",
    "www.ftc.go.kr",
    "kcabinternational.or.kr",
    "www.kcabinternational.or.kr",
}


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL is not on the official-site allowlist: {url}")


class OfficialSiteRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        validate_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url,
        )


def validated_destination(relative_path: str) -> Path:
    destination = (PROJECT_ROOT / relative_path).resolve()
    try:
        destination.relative_to(DOCUMENTS_PATH)
    except ValueError as error:
        raise ValueError(f"Download destination is outside documents/: {relative_path}") from error
    return destination


def validate_file(destination: Path, content: bytes) -> None:
    suffix = destination.suffix.casefold()
    signatures = {
        ".pdf": b"%PDF-",
        ".hwpx": b"PK",
        ".zip": b"PK",
        ".hwp": bytes.fromhex("D0CF11E0A1B11AE1"),
    }
    signature = signatures.get(suffix)
    if signature and not content.startswith(signature):
        raise ValueError(f"Downloaded content is not a valid {suffix} file: {destination.name}")


def fetch(request: Request, opener) -> tuple[bytes, str]:
    with opener.open(request, timeout=60) as response:
        validate_url(response.geturl())
        return response.read(), response.geturl()


def save_content(relative_path: str, content: bytes) -> str:
    destination = validated_destination(relative_path)
    if destination.is_file():
        return f"SKIP {relative_path}"
    validate_file(destination, content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return f"SAVE {relative_path} ({len(content):,} bytes)"


def download(entry: dict, opener) -> str:
    url = entry["url"]
    validate_url(url)
    destination = validated_destination(entry["path"])
    if destination.is_file():
        return f"SKIP {entry['path']}"

    request_body = None
    if "form" in entry:
        request_body = urlencode(entry["form"], doseq=True).encode("utf-8")
    request = Request(
        url,
        data=request_body,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    content, _ = fetch(request, opener)
    return save_content(entry["path"], content)


def html_value(page: str, element_id: str) -> str:
    pattern = rf'id="{re.escape(element_id)}"[^>]*value="([^"]*)"'
    match = re.search(pattern, page)
    if not match:
        raise ValueError(f"Official law page is missing {element_id}")
    return unescape(match.group(1))


def download_law(entry: dict, opener) -> list[str]:
    name = entry["law_name"]
    current_date = date.today().strftime("%Y%m%d")
    landing_url = entry.get(
        "source_url",
        f"https://www.law.go.kr/{quote('법령')}/{quote(name)}?efYd={current_date}",
    )
    page_bytes, page_url = fetch(Request(landing_url, headers={"User-Agent": "Mozilla/5.0"}), opener)
    page = page_bytes.decode("utf-8", errors="replace")
    if 'id="lsiSeq"' not in page:
        frame = re.search(r'<iframe[^>]+src="([^"]+)"', page)
        if not frame:
            raise ValueError(f"Could not locate the official text for {name}")
        inner_url = urljoin(page_url, unescape(frame.group(1)))
        validate_url(inner_url)
        page_bytes, _ = fetch(Request(inner_url, headers={"User-Agent": "Mozilla/5.0"}), opener)
        page = page_bytes.decode("utf-8", errors="replace")

    lsi_seq = html_value(page, "lsiSeq")
    official_name = html_value(page, "lsNm")
    enactment_number = html_value(page, "ancNo")
    character_class = html_value(page, "lsBdyChrCls")
    effective_match = re.search(r"var efYd = '(\d{8})'", page)
    type_match = re.search(r"\[([^\]\s]+) \uc81c\d+\ud638,", page)
    if not effective_match or not type_match:
        raise ValueError(f"Could not read version metadata for {name}")
    effective_date = entry.get("effective_date", effective_match.group(1))
    law_type = type_match.group(1)
    base_name = f"{official_name}({law_type})(\uc81c{enactment_number}\ud638)({effective_date})"
    directory = entry["directory"].rstrip("/")

    common_form = {
        "lsiSeq": lsi_seq,
        "chrClsCd": character_class,
        "joAllCheck": "Y",
        "joEfOutPutYn": "on",
        "mokChaChk": "Y",
        "bylChaChk": "Y",
        "efYd": effective_date,
        "lsJoSize": "10",
        "lsNmFont": "goThic",
        "lsJoFont": "smyoungjo",
        "spaceCls": "2",
    }
    hwpx_url = (
        "https://www.law.go.kr/LSW/lsHwpxSave.do?"
        + urlencode({"trSeq": lsi_seq, "lastCheck": "Y", "ancYnChk": "0"})
    )
    hwpx_request = Request(
        hwpx_url,
        data=urlencode(common_form).encode("utf-8"),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    hwpx_content, _ = fetch(hwpx_request, opener)

    pdf_query = dict(common_form)
    pdf_query["ancYnChk"] = "0"
    pdf_url = "https://www.law.go.kr/LSW/lsPdfPrint.do?" + urlencode(pdf_query)
    pdf_content, _ = fetch(Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"}), opener)
    return [
        save_content(f"{directory}/{base_name}.hwpx", hwpx_content),
        save_content(f"{directory}/{base_name}.pdf", pdf_content),
    ]


def download_administrative_rule(entry: dict, opener) -> list[str]:
    sequence = entry["sequence"]
    character_class = entry["character_class"]
    list_url = "https://www.law.go.kr/LSW/admRulJoListRInc.do?" + urlencode(
        {"admRulSeq": sequence, "mode": "99", "chapNo": "1", "nwYn": "1"}
    )
    list_content, _ = fetch(Request(list_url, headers={"User-Agent": "Mozilla/5.0"}), opener)
    articles = json.loads(list_content)
    article_numbers = []
    for article in articles:
        if article.get("joYn") == "Y":
            article_numbers.append(
                f"{article['oriJoNo']}-{article.get('dashNo', '')}:{article['joBrNo']}"
            )
        elif article.get("allYn"):
            article_numbers.append(article["chapNo"])
    if not article_numbers:
        raise ValueError(f"Official article list is empty for {entry['name']}")

    if any(article.get("allYn") for article in articles):
        body_request = Request(
            "https://www.law.go.kr/LSW/admRulInfoR.do",
            data=urlencode(
                {
                    "admRulSeq": sequence,
                    "admRulId": entry["administrative_rule_id"],
                    "joTpYn": "N",
                    "languageType": "KO",
                    "chrClsCd": character_class,
                }
            ).encode("utf-8"),
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )
        body_content, _ = fetch(body_request, opener)
        body_html = body_content.decode("utf-8", errors="replace")
        content = body_html.split('<div id="contentBody">', 1)[-1].split("<script", 1)[0]
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        content = re.sub(r"</(?:p|h\d|div|li)>|<br\s*/?>", "\n", content)
        content = unescape(re.sub(r"<[^>]+>", "", content))
        lines = []
        for line in content.splitlines():
            stripped_line = line.strip()
            if stripped_line:
                lines.append(stripped_line)
        text_content = ("\n".join(lines) + "\n").encode("utf-8")
        path = f"{entry['directory']}/{entry['name']}.txt"
        return [save_content(path, text_content)]

    form = {
        "admRulSeq": sequence,
        "joNo": ",".join(article_numbers),
        "chrClsCd": character_class,
        "mokChaChk": "Y",
        "bylChaChk": "Y",
        "allJoChkYn": "Y",
    }
    results = []
    for suffix, endpoint in (
        ("pdf", "admRulPdfPrint.do"),
        ("hwpx", "admRulHwpxSave.do"),
    ):
        url = f"https://www.law.go.kr/LSW/{endpoint}"
        request = Request(
            url,
            data=urlencode(form).encode("utf-8"),
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )
        content, _ = fetch(request, opener)
        results.append(save_content(f"{entry['directory']}/{entry['name']}.{suffix}", content))
    return results


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    opener = build_opener(OfficialSiteRedirects())
    direct_only = "--direct-only" in sys.argv[1:]
    failures = 0
    for entry in plan:
        if direct_only and ("law_name" in entry or "administrative_rule" in entry):
            continue
        try:
            if "law_name" in entry:
                messages = download_law(entry, opener)
            elif "administrative_rule" in entry:
                messages = download_administrative_rule(entry, opener)
            else:
                messages = [download(entry, opener)]
            for message in messages:
                print(message)
        except Exception as error:
            failures += 1
            label = entry.get("path", entry.get("law_name", "<missing entry>"))
            print(f"FAIL {label}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
