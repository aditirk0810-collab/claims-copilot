"""Deep gold questions: each needs 2+ documents, and often math or a comparison.

Answers are computed from the claim facts, so ground truth is exact. `evidence_pages`
lists every page needed (the agent must look at several documents to answer).

  answer_kind: "money" | "days" | "yes_no" | "flags"
"""

from __future__ import annotations

import random

from .claim import Claim, money


def build_deep_qa(cl: Claim, fact_pages: dict[str, list[int]], rng: random.Random) -> list[dict]:
    qa: list[dict] = []

    def pages(*keys: str) -> list[int]:
        return sorted({p for k in keys for p in fact_pages.get(k, [])})

    def add(qtype: str, question: str, answer: str, kind: str, keys: list[str], docs: int) -> None:
        qa.append({
            "id": f"{cl.claim_id}/d{len(qa) + 1:02d}",
            "claim_id": cl.claim_id,
            "type": qtype,
            "question": question,
            "answer": answer,
            "answer_kind": kind,
            "evidence_pages": pages(*keys),
            "documents_needed": docs,
        })

    est_keys = [f"estimate_total:{i}" for i in range(len(cl.estimates))]
    low = cl.lowest_estimate

    # 1. Total exposure: lower estimate + medical + towing (3-4 documents, addition).
    total = low.total + (cl.medical.total if cl.medical else 0) + (cl.towing.amount if cl.towing else 0)
    keys = est_keys + (["medical_total"] if cl.medical else []) + (["tow_amount"] if cl.towing else [])
    add("total_exposure",
        rng.choice(["What is the total known cost of this claim, using the lower repair estimate plus any medical bills and towing?",
                    "Add up the claim's costs: the cheaper repair estimate, all medical charges, and towing. What is the total?"]),
        money(total), "money", keys, len(cl.estimates) + bool(cl.medical) + bool(cl.towing))

    # 2. Gap between two estimates (2 documents, subtraction).
    if len(cl.estimates) == 2:
        gap = abs(cl.estimates[0].total - cl.estimates[1].total)
        add("estimate_gap", "How much more expensive is the higher repair estimate than the lower one?",
            money(gap), "money", est_keys, 2)

    # 3. Days from accident to treatment (loss notice + medical bill, date math).
    if cl.medical and "treatment_before_loss" not in cl.anomalies:
        days = (cl.medical.date_of_service - cl.date_of_loss).days
        add("days_to_treatment", "How many days after the accident did the claimant receive medical treatment?",
            str(days), "days", ["date_of_loss", "date_of_service"], 2)

    # 4. Days from loss to adjuster acknowledgement (loss notice + email, date math).
    add("days_to_ack", "How many days after the date of loss did the adjuster acknowledge the claim?",
        str((cl.ack_date - cl.date_of_loss).days), "days", ["date_of_loss", "ack_date"], 2)

    # 5. Was OUR policyholder at fault? (police report + loss notice, compare names).
    if cl.police:
        add("claimant_at_fault", "According to the police report, was our policyholder the at-fault driver?",
            "Yes" if cl.police.at_fault == cl.claimant.name else "No", "yes_no",
            ["at_fault_driver", "policy_number"], 2)

    # 6. Do the loss notice and police report agree on the accident date?
    if cl.police:
        add("dates_consistent", "Do the loss notice and the police report agree on the date of the accident?",
            "No" if "police_date_mismatch" in cl.anomalies else "Yes", "yes_no",
            ["date_of_loss", "police_date"], 2)

    # 7. Injury answer on the loss notice vs. medical documents in the packet.
    add("injury_consistent",
        "Is the injury answer on the loss notice consistent with the medical documents in this packet?",
        "No" if "injury_mismatch" in cl.anomalies else "Yes", "yes_no",
        ["injury_reported"] + (["medical_total"] if cl.medical else []), 2 if cl.medical else 1)

    # 8. Same vehicle on every estimate as on the loss notice?
    add("vehicle_consistent", "Does every repair estimate list the same vehicle as the loss notice?",
        "No" if "vehicle_mismatch" in cl.anomalies else "Yes", "yes_no",
        ["vehicle"] + [f"estimate_vehicle:{i}" for i in range(len(cl.estimates))], 1 + len(cl.estimates))

    # 9. Treatment date before the accident? (fraud red flag)
    if cl.medical:
        add("treatment_before_accident", "Was any medical treatment dated before the accident happened?",
            "Yes" if "treatment_before_loss" in cl.anomalies else "No", "yes_no",
            ["date_of_loss", "date_of_service"], 2)

    # 10. Open review: list every inconsistency (none, or the planted one).
    add("red_flags", "Review the whole packet. List any inconsistencies between the documents, or say there are none.",
        ",".join(cl.anomalies) or "none", "flags",
        ["date_of_loss", "injury_reported", "vehicle"] + est_keys
        + (["police_date"] if cl.police else []) + (["date_of_service", "medical_total"] if cl.medical else []), 4)
    return qa
