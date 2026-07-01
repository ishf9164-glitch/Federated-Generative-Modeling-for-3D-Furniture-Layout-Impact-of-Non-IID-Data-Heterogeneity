import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


ROLE_LABELS = {
    "user": "用户",
    "assistant": "Codex",
    "tool": "工具",
}


def iso_to_local_text(value):
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def content_to_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text:
            parts.append(text)
            continue
        if item.get("type") in {"input_text", "output_text"} and item.get("text"):
            parts.append(item["text"])
    return "\n\n".join(parts).strip()


def compact_tool_payload(payload):
    ptype = payload.get("type", "")
    if ptype == "function_call":
        name = payload.get("name") or payload.get("call_id") or "tool"
        args = payload.get("arguments")
        return f"调用工具: {name}\n{args or ''}".strip()
    if ptype == "function_call_output":
        output = payload.get("output", "")
        return str(output).strip()
    if ptype in {"local_shell_call", "local_shell_call_output"}:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return ""


def iter_entries(source):
    session_meta = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")
            payload = record.get("payload") or {}
            timestamp = iso_to_local_text(record.get("timestamp"))

            if rtype == "session_meta":
                session_meta = payload
                continue

            if rtype != "response_item" or not isinstance(payload, dict):
                continue

            ptype = payload.get("type")
            if ptype == "message":
                role = payload.get("role", "")
                if role not in {"user", "assistant"}:
                    continue
                text = content_to_text(payload.get("content"))
                if not text:
                    continue
                yield {
                    "timestamp": timestamp,
                    "label": ROLE_LABELS.get(role, role or "消息"),
                    "text": text,
                    "line_number": line_number,
                }
            else:
                text = compact_tool_payload(payload)
                if text:
                    yield {
                        "timestamp": timestamp,
                        "label": "工具",
                        "text": text,
                        "line_number": line_number,
                    }

    return session_meta


def split_paragraphs(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n{2,}", text)
    paragraphs = []
    for chunk in chunks:
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        paragraphs.append(chunk)
    return paragraphs or [""]


def paragraph_xml(text, style=None):
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    lines = text.split("\n")
    runs = []
    for idx, line in enumerate(lines):
        if idx:
            runs.append("<w:r><w:br/></w:r>")
        preserve = ' xml:space="preserve"' if line[:1].isspace() or line[-1:].isspace() else ""
        runs.append(f"<w:r><w:t{preserve}>{escape(line)}</w:t></w:r>")
    return f"<w:p>{style_xml}{''.join(runs)}</w:p>"


def make_document_xml(title, subtitle, entries):
    body = [
        paragraph_xml(title, "Title"),
        paragraph_xml(subtitle, "Subtitle"),
    ]

    for idx, entry in enumerate(entries, 1):
        heading = f"{idx}. {entry['label']}"
        if entry["timestamp"]:
            heading += f"  {entry['timestamp']}"
        body.append(paragraph_xml(heading, "Heading1"))
        for para in split_paragraphs(entry["text"]):
            body.append(paragraph_xml(para))

    body.append(
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar '
        'w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}</w:body>"
        "</w:document>"
    )


def make_styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:qFormat/>
    <w:rPr><w:color w:val="666666"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="300" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
  </w:style>
</w:styles>"""


def write_docx(output, document_xml):
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", make_styles_xml())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    entries = list(iter_entries(args.source))
    if not entries:
        raise SystemExit("No user/assistant conversation entries were found.")

    title = "Codex 会话记录"
    subtitle = f"来源: {args.source.name} | 条目: {len(entries)} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    document_xml = make_document_xml(title, subtitle, entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_docx(args.output, document_xml)
    print(args.output)
    print(f"entries={len(entries)}")


if __name__ == "__main__":
    main()
