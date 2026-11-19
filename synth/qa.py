"""Gold questions for a packet, each with the exact answer and the evidence pages.

`fact_pages` maps a fact key to the final (shuffled) page numbers it is printed on,
so every answer cites real pages. Questions about documents that are missing from
the packet get the answer NOT_IN_PACKET: the copilot must say so instead of guessing.
"""

from __future__ import annotations

import random

from .claim import Claim, money

NOT_IN_PACKET = "NOT_IN_PACKET"


def build_qa(cl: Claim, fact_pages: dict[str, list[int]], rng: random.Random) -> list[dict]:
    qa: list[dict] = []

    def add(qtype: str, phrasings: list[str], answer: str, keys: list[str]) -> None:
        evidence = sorted({p for k in keys for p in fact_pages.get(k, [])})
        if answer != NOT_IN_PACKET and not evidence:
            raise ValueError(f"{cl.claim_id}: no evidence page for {qtype}")
        qa.append({
            "id": f"{cl.claim_id}/q{len(qa) + 1:02d}",
            "claim_id": cl.claim_id,
            "type": qtype,
            "question": rng.choice(phrasings),
            "answer": answer,
            "evidence_pages": evidence,
            "answerable": answer != NOT_IN_PACKET,
        })

    add("policy_number", ["What is the policy number?", "Which policy is this claim filed under?"],
        cl.policy_number, ["policy_number"])
    add("date_of_loss", ["What is the date of loss?", "When did the accident happen?"],
        cl.date_of_loss.strftime("%m/%d/%Y"), ["date_of_loss"])
    add("vehicle", ["What vehicle was involved?", "Which car does the claim cover?"],
        cl.vehicle.label, ["vehicle"])
    add("claimant_phone", ["What is the claimant's phone number?"], cl.claimant.phone, ["claimant_phone"])
    add("injury_reported", ["Did the claimant report any injuries?", "Were injuries reported on the loss notice?"],
        "Yes" if cl.injury else "No", ["injury_reported"])
    add("adjuster", ["Who is the assigned adjuster?"], cl.adjuster, ["adjuster"])
    add("ack_date", ["When did the adjuster acknowledge the claim?"],
        cl.ack_date.strftime("%m/%d/%Y"), ["ack_date"])

    if cl.police:
        add("police_report_number", ["What is the police report number?", "Is there a police report? Give the number."],
            cl.police.report_no, ["police_report_number"])
        add("at_fault_driver", ["Who was at fault according to the police report?"],
            cl.police.at_fault, ["at_fault_driver"])
    else:
        add("police_report_number", ["What is the police report number?", "Is there a police report? Give the number."],
            NOT_IN_PACKET, [])

    for i, e in enumerate(cl.estimates):
        add("estimate_total", [f"What is the total of the {e.shop} repair estimate?",
                               f"How much did {e.shop} quote for the repair?"],
            money(e.total), [f"estimate_total:{i}"])
    if len(cl.estimates) > 1:
        add("lowest_estimate", ["Which repair shop gave the lower estimate?"],
            cl.lowest_estimate.shop, [f"estimate_total:{i}" for i in range(len(cl.estimates))])

    if cl.medical:
        add("medical_total", ["What is the total medical bill amount?", "How much are the medical charges?"],
            money(cl.medical.total), ["medical_total"])
        add("diagnosis", ["What was the diagnosis on the medical bill?"], cl.medical.icd[1], ["diagnosis"])
    else:
        add("medical_total", ["What is the total medical bill amount?", "How much are the medical charges?"],
            NOT_IN_PACKET, [])

    if cl.towing:
        add("tow_amount", ["How much was the towing charge?"], money(cl.towing.amount), ["tow_amount"])
    else:
        add("tow_amount", ["How much was the towing charge?"], NOT_IN_PACKET, [])

    return qa
