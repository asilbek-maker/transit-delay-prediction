"""Minimal standard-library .xlsx reader (first sheet -> list of rows).

Avoids an openpyxl dependency for the one Seoul file we need.
"""
import re, sys, zipfile
import xml.etree.ElementTree as ET
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

def read_rows(path):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter(f"{{{NS['m']}}}row"):
        vals = {}
        for c in row.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", NS)
            text = (v.text if v is not None else "") or ""
            if c.get("t") == "s":
                text = shared[int(text)]
            elif c.get("t") == "inlineStr":
                text = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
            vals[col] = text
        cols = sorted(vals, key=lambda k: (len(k), k))
        rows.append([vals.get(k, "") for k in cols])
    return rows

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = read_rows(sys.argv[1])
    print(len(rows), "rows")
    for r in rows[:4]:
        print(r)
