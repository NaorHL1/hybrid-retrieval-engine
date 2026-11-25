"""Draft helper for the identifier bucket.
For each (query, target, needle): lists every doc whose text literally contains `needle`
(exact-match pool, catches duplicates dense/bm25 might miss) and the target's rank in dense and bm25.
"""
import sys
sys.path.insert(0, ".")
from retrieval import open_table, dense, bm25

DRAFT = [
 ("A6-FKF", "doc_0003", "A6-FKF"),
 ("+971 54 232 3578", "doc_0077", "232 3578"),
 ("transponder code 7700", "doc_0118", "7700"),
 ("FZ1081", "doc_0134", "FZ1081"),
 ("Air Transport Law 1980", "doc_0104", "Air Transport Law"),
 ("Iberia Airlines v. Fleischer Peled", "doc_0106", "Fleischer"),
 ("Victims of Hostile Actions (Pensions) Law, 1970", "doc_0115", "Hostile Actions"),
 ("Montreal Convention", "doc_0108", "Montreal Convention"),
 ("Ghaith Al Ghaith", "doc_0153", "Ghaith"),
 ("Sally Leivesley", "doc_0028", "Leivesley"),
 ("Richard J. Levy", "doc_0027", "Levy"),
 ("Royal Air Maroc", "doc_0145", "Royal Air Maroc"),
 ("Humam Khalil Abu-Mulal al-Balawi", "doc_0016", "Balawi"),
 ("7,479 passengers", "doc_0152", "7,479"),
 ("10:45 a.m. GST (9:45 a.m. SAST)", "doc_0011", "10:45"),
]
t = open_table()
n = t.count_rows()
docs = {r["id"]: r["text"] for r in t.search().select(["id","text"]).limit(n).to_list()}
for i, (q, target, needle) in enumerate(DRAFT, 1):
    d = [x["id"] for x in dense(t, q, n)]
    b = [x["id"] for x in bm25(t, q, n)]
    rd = d.index(target)+1 if target in d else None
    rb = b.index(target)+1 if target in b else None
    hits = [k for k, v in docs.items() if needle.lower() in v.lower()]
    print(f"\n#{i} Q={q!r} target={target} | rank dense={rd} bm25={rb} (bm25 returned {len(b)})")
    print(f"   TARGET: {docs[target][:200]!r}")
    for h in hits:
        if h == target: continue
        j = docs[h].lower().find(needle.lower())
        print(f"   contains {needle!r} -> {h}: ...{docs[h][max(0,j-60):j+90]!r}")
