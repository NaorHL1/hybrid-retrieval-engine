"""Ground truth for the retrieval eval.

type:
  semantic   - paraphrase / meaning, little exact wording overlap  (dense should shine)
  identifier - rare exact token: ID, name, number, phone           (BM25 should shine)
  mixed      - natural question that also carries an exact token

relevant: ALL doc ids that legitimately answer the query. Several docs can answer the same
question, so a list is fairer than one id.
"""

EVAL_SET = [
    # ---- carried over from the original benchmark.py ----
    ("what al-Hammami did before the attack?", ["doc_0094"], "semantic"),
    ("Who was the medical professional that treated the wounded pilot?", ["doc_0010"], "semantic"),
    ("How did a passenger know how to stabilize the plane controls from a TV show?", ["doc_0085"], "semantic"),
    ("Why did the attacker abort a previous hijacking attempt on an earlier flight?", ["doc_0051"], "semantic"),
    ("Squawk 7500 and 7700 emergency broadcast", ["doc_0008"], "mixed"),
    ("Exactly 2,963 out of 4,136 social media reports contained false flags", ["doc_0026"], "mixed"),
    ("What is the maximum liability in special drawing rights equivalent to $210,000?", ["doc_0100"], "mixed"),
    ("Did the suspect undergo radicalization while on a student visa in Australia between 2015 and 2017?", ["doc_0017"], "mixed"),
    ("The 1976 Air France Flight 139 hijacking with a stop in Athens", ["doc_0112"], "mixed"),
    ("How many Next-Generation Boeing 737-800 and Boeing 737 MAX 8 aircraft are operated?", ["doc_0076"], "mixed"),
    ("A6-FKF", ["doc_0003"], "identifier"),
    ("Aviation Experts Consulting", ["doc_0027"], "identifier"),
    ("Shin Bet", ["doc_0036"], "identifier"),
    ("Tzvika Manes", ["doc_0009"], "identifier"),
    ("Air Transport Law 1980", ["doc_0104"], "identifier"),
    ("+971 54 232 3578", ["doc_0077"], "identifier"),
    ("10:45 a.m. GST (9:45 a.m. SAST)", ["doc_0011"], "identifier"),
    ("legislative election on 27 October", ["doc_0007"], "identifier"),
    # doc_0106 describes the Iberia case, doc_0107 names it (Iberia Airlines v. Fleischer Peled): both are valid.
    ("Iberia Airlines v. Fleischer Peled", ["doc_0106", "doc_0107"], "identifier"),
    ("Dunedin, New Zealand SpiceJet 11 years", ["doc_0004"], "mixed"),
    # ---- new (written from the doc texts; check them, and ADD YOUR OWN) ----
    ("Which country refused to let Israeli military planes pick up the stranded passengers?", ["doc_0020"], "semantic"),
    ("How long did Israel ban Flydubai flights after the incident?", ["doc_0022"], "semantic"),
    ("How close was the aircraft to a dive it could not recover from?", ["doc_0028"], "semantic"),
    ("Did anyone else help the suspect or was it a lone attacker?", ["doc_0039", "doc_0098"], "semantic"),
    ("Can passengers claim compensation for psychological trauma after a hijacking?", ["doc_0103", "doc_0107"], "semantic"),
    ("Why was the airline's background check on the pilot insufficient?", ["doc_0018", "doc_0055"], "semantic"),
    ("Anwar Gargash", ["doc_0024"], "identifier"),
    ("Victims of Hostile Actions (Pensions) Law, 1970", ["doc_0115"], "identifier"),
    ("151,880 special drawing rights", ["doc_0100"], "identifier"),
    ("Ayman al-Zawahiri", ["doc_0016"], "identifier"),
    ("Who called the incident an act of terrorism first among Emirati officials, Gargash?", ["doc_0024"], "mixed"),
    ("Which airlines resumed flights to Israel after UAE agreed to security demands: El Al, Arkia, Israir?", ["doc_0022"], "mixed"),
]
