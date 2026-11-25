"""Draft helper for the mixed bucket: identifier shared by several docs + free words that disambiguate."""
import sys
sys.path.insert(0, ".")
from retrieval import open_table, dense, bm25

DRAFT = [
 ("Under the Montreal Convention, what is the maximum an airline owes each passenger?", "doc_0100", "Montreal Convention"),
 ("Can passengers sue flydubai in Israeli courts under the Montreal Convention?", "doc_0108", "Montreal Convention"),
 ("Does the Montreal Convention allow compensation for emotional distress alone?", "doc_0106", "Montreal Convention"),
 ("Which pilots does the Jerusalem Post say reached the cockpit?", "doc_0141", "Jerusalem Post"),
 ("What did Oman Air do with al-Hammami before he left?", "doc_0014", "Oman Air"),
 ("At what local time did the Tabuk airports cluster receive the distress call?", "doc_0138", "Tabuk"),
 ("What did Attorney-General al-Shamsi announce on 3 October?", "doc_0130", "Shamsi"),
 ("When did the transponder code change to 7500 and what does it signal?", "doc_0118", "7500"),
 ("How high did FZ1073 climb after departing Dubai according to Flightradar24?", "doc_0117", "FZ1073"),
 ("Which lawyer named Regev represented a couple against a Spanish airline?", "doc_0106", "Regev"),
 ("What did transport minister Regev say about passenger lists?", "doc_0058", "Regev"),
 ("Which airline stopped its codeshare with flydubai to Tel Aviv?", "doc_0151", "codeshare"),
 ("What did Captain Machchhar tell Netanyahu about the co-pilot?", "doc_0094", "Machchhar"),
 ("How did Gargash describe the captain's actions?", "doc_0135", "Gargash"),
 ("What did the Saudi National Transport Safety Center do at Tabuk?", "doc_0140", "Tabuk"),
]
t = open_table(); n = t.count_rows()
docs = {r["id"]: r["text"] for r in t.search().select(["id","text"]).limit(n).to_list()}
for i, (q, target, needle) in enumerate(DRAFT, 1):
    d = [x["id"] for x in dense(t, q, n)]; b = [x["id"] for x in bm25(t, q, n)]
    rd = d.index(target)+1 if target in d else None; rb = b.index(target)+1 if target in b else None
    nd = [k for k, v in docs.items() if needle.lower() in v.lower()]
    print(f"\n#{i} Q={q}\n   target={target} dense={rd} bm25={rb} | docs containing {needle!r}: {len(nd)}")
    print(f"   TARGET: {docs[target][:260]!r}")
    pool = []
    for h in d[:5] + b[:5]:
        if h not in pool and h != target: pool.append(h)
    for h in pool: print(f"   cand {h}: {docs[h][:100]!r}")
