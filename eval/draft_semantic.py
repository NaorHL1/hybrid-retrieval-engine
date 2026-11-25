"""Draft helper: prints each draft query, its intended target, and pooled candidates.
Pool = top-5 dense UNION top-5 bm25 (cross-encoder intentionally excluded, to avoid pooling bias toward it).
"""
import json, sys
sys.path.insert(0, ".")
from retrieval import open_table, dense, bm25

DRAFT = [
 ("doc_0012", "Who took the suspect into custody after the plane touched down, and where were the wounded pilots treated?"),
 ("doc_0020", "Why was Israel unable to send its own military planes to bring its citizens home?"),
 ("doc_0025", "Does the government itself check the backgrounds of pilots employed by overseas airlines?"),
 ("doc_0028", "How close did the jet come to hitting the ground before it recovered?"),
 ("doc_0031", "What facility did the accused pilot intend to strike?"),
 ("doc_0038", "Why should a crew member from a country without relations with Israel have been stopped at the border?"),
 ("doc_0054", "What did his previous employer do when worried about his extremist leanings?"),
 ("doc_0132", "How did other carriers respond commercially after the attack?"),
 ("doc_0126", "How did the travelers eventually finish their trip after the emergency landing?"),
 ("doc_0143", "How quickly did the aircraft lose height during the struggle in the cockpit?"),
 ("doc_0150", "Did prosecutors say he had accomplices?"),
 ("doc_0149", "What do forensic findings say about the injuries of the wounded commander?"),
 ("doc_0152", "How many people had flown between the two cities in the days before the incident?"),
 ("doc_0125", "Which foreign head of government spoke to the injured commander from his hospital room?"),
 ("doc_0147", "Had he rehearsed the route on earlier trips before the attack?"),
]
t = open_table()
docs = {r["id"]: r for r in t.search().select(["id","title","text"]).limit(t.count_rows()).to_list()}
out = []
for i, (target, q) in enumerate(DRAFT, 1):
    pool = []
    for hit in dense(t, q, 5) + bm25(t, q, 5):
        if hit["id"] not in pool: pool.append(hit["id"])
    print(f"\n#{i} Q: {q}\n   TARGET {target}: {docs[target]['text'][:230]!r}")
    for pid in pool:
        if pid == target: continue
        print(f"   cand {pid}: {docs[pid]['text'][:120]!r}")
    out.append({"id": f"sem-{i:02d}", "bucket": "semantic", "query": q, "expected_id": target, "also_relevant": [], "verified": False})
json.dump(out, open("eval/_draft_semantic.json", "w"), indent=1)
