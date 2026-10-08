import lancedb
import asyncio
import time

from data_models import FlightDocument
from api import search_incident, hybrid_search_incident

ground_truth = {
    "what al-Hammami did before the attack?": "doc_0094",
    "Who was the medical professional that treated the wounded pilot?": "doc_0010",
    "How did a passenger know how to stabilize the plane controls from a TV show?": "doc_0085",
    "Why did the attacker abort a previous hijacking attempt on an earlier flight?": "doc_0051",
    "Squawk 7500 and 7700 emergency broadcast": "doc_0008",
    "Exactly 2,963 out of 4,136 social media reports contained false flags": "doc_0026",
    "What is the maximum liability in special drawing rights equivalent to $210,000?": "doc_0100",
    "Did the suspect undergo radicalization while on a student visa in Australia between 2015 and 2017?": "doc_0017",
    "The 1976 Air France Flight 139 hijacking with a stop in Athens": "doc_0112",
    "How many Next-Generation Boeing 737-800 and Boeing 737 MAX 8 aircraft are operated?": "doc_0076",
    "A6-FKF": "doc_0003",
    "Aviation Experts Consulting": "doc_0027",
    "Shin Bet": "doc_0036",
    "Tzvika Manes": "doc_0009",
    "Air Transport Law 1980": "doc_0104",
    "+971 54 232 3578": "doc_0077",
    "10:45 a.m. GST (9:45 a.m. SAST)": "doc_0011",
    "legislative election on 27 October": "doc_0007",
    "Iberia Airlines v. Fleischer Peled": "doc_0106",
    "Dunedin, New Zealand SpiceJet 11 years": "doc_0004",
}


async def main():

    vec_search_counter = 0
    hyb_search_counter = 0

    total_vec_time = 0
    total_hyb_time = 0

    for query, expected_id in ground_truth.items():

        start_vec = time.time()
        vec_results = await search_incident(query)
        end_vec = time.time()
        total_vec_time += end_vec - start_vec

        top_3_vec_results = vec_results[:3]

        vec_ids = [doc["id"] for doc in top_3_vec_results]

        if expected_id in vec_ids:
            vec_search_counter += 1
        else:
            print("Vec fail: ", query)

        start_hyb = time.time()
        hyb_results = await hybrid_search_incident(query)
        end_hyb = time.time()
        total_hyb_time += end_hyb - start_hyb

        top_3_hyb_results = hyb_results[:3]

        hyb_ids = [doc["id"] for doc in top_3_hyb_results]

        if expected_id in hyb_ids:
            hyb_search_counter += 1
        else:
            print("Hyb fail: ", query)

    # ------------------ PRINT REPORT ------------------
    total_queries = len(ground_truth)
    avg_vec_time = total_vec_time / total_queries
    avg_hyb_time = total_hyb_time / total_queries

    vec_accuracy = (vec_search_counter / total_queries) * 100
    hyb_accuracy = (hyb_search_counter / total_queries) * 100
    print("\n" + "=" * 50)
    print("      BENCHMARK RESULTS (Hit Rate@3)")
    print("=" * 50)

    print("\n[ PURE VECTOR SEARCH ]")
    print(
        f"Accuracy:    {vec_accuracy:.1f}% ({vec_search_counter}/{total_queries} Hits)"
    )
    print(f"Total Time:  {total_vec_time:.3f}s")
    print(f"Avg Latency: {avg_vec_time:.3f}s per query")

    print("\n[ HYBRID SEARCH (Vector + BM25 + RRF) ]")
    print(
        f"Accuracy:    {hyb_accuracy:.1f}% ({hyb_search_counter}/{total_queries} Hits)"
    )
    print(f"Total Time:  {total_hyb_time:.3f}s")
    print(f"Avg Latency: {avg_hyb_time:.3f}s per query")
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":

    asyncio.run(main())
