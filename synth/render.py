"""Render each document of a claim packet as single PDF pages.

Each renderer returns `RenderedPage`s carrying the page type label (for the
classifier) and the facts printed on that page (for gold Q&A citations).
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

from .claim import DATE_FMT, INSURER, INSURER_ADDRESS, Claim, money

W, H = letter
MARGIN = 54

PAGE_TYPES = [
    "claim_form",
    "police_report",
    "repair_estimate",
    "medical_bill",
    "correspondence",
    "photo",
    "other",
]


@dataclass
class RenderedPage:
    doc_id: str
    type: str
    pdf: bytes
    facts: dict[str, str] = field(default_factory=dict)  # fact key -> exact printed answer
    source: str | None = None  # photo pages: source image file, or "drawn"


# ---------- drawing helpers ----------

class Page:
    def __init__(self) -> None:
        self.buf = io.BytesIO()
        self.c = canvas.Canvas(self.buf, pagesize=letter)
        self.y = H - MARGIN

    def bytes(self) -> bytes:
        self.c.showPage()
        self.c.save()
        return self.buf.getvalue()

    def banner(self, title: str, org: str, org_line: str = "") -> None:
        c = self.c
        c.setFillGray(0.15)
        c.rect(0, H - 70, W, 70, fill=1, stroke=0)
        c.setFillGray(1)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(MARGIN, H - 34, org)
        c.setFont("Helvetica", 9)
        if org_line:
            c.drawString(MARGIN, H - 50, org_line)
        c.setFillGray(0)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(MARGIN, H - 98, title)
        self.y = H - 122

    def kv(self, label: str, value: str, x: float = MARGIN, step: bool = True) -> None:
        self.c.setFont("Helvetica-Bold", 9.5)
        self.c.drawString(x, self.y, f"{label}:")
        self.c.setFont("Helvetica", 9.5)
        self.c.drawString(x + 120, self.y, value)
        if step:
            self.y -= 16

    def kv2(self, left: tuple[str, str], right: tuple[str, str]) -> None:
        self.kv(*left, step=False)
        self.kv(*right, x=W / 2 + 10)

    def section(self, title: str) -> None:
        self.y -= 6
        self.c.setFont("Helvetica-Bold", 10.5)
        self.c.drawString(MARGIN, self.y, title.upper())
        self.c.setLineWidth(0.6)
        self.c.line(MARGIN, self.y - 4, W - MARGIN, self.y - 4)
        self.y -= 20

    def para(self, text: str, size: float = 10, leading: float = 14) -> None:
        self.c.setFont("Helvetica", size)
        for line in simpleSplit(text, "Helvetica", size, W - 2 * MARGIN):
            self.c.drawString(MARGIN, self.y, line)
            self.y -= leading
        self.y -= 6

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        c = self.c
        xs = [MARGIN]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)
        c.setFont("Helvetica-Bold", 9)
        for x, h, w in zip(xs, headers, widths):
            c.drawString(x, self.y, h) if h != headers[-1] else c.drawRightString(x + w, self.y, h)
        self.y -= 5
        c.line(MARGIN, self.y, W - MARGIN, self.y)
        self.y -= 13
        c.setFont("Helvetica", 9)
        for row in rows:
            for i, (x, v, w) in enumerate(zip(xs, row, widths)):
                if i == len(row) - 1:
                    c.drawRightString(x + w, self.y, v)
                else:
                    c.drawString(x, self.y, v)
            self.y -= 14
        self.y -= 4

    def total_line(self, label: str, value: str, bold: bool = False) -> None:
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", 10 if bold else 9.5)
        self.c.drawRightString(W - MARGIN - 110, self.y, label)
        self.c.drawRightString(W - MARGIN, self.y, value)
        self.y -= 15


def _d(d) -> str:
    return d.strftime(DATE_FMT)


# ---------- documents ----------

def claim_form(cl: Claim) -> list[RenderedPage]:
    p = Page()
    p.banner("Automobile Loss Notice - First Notice of Loss", INSURER, INSURER_ADDRESS)
    p.kv2(("Claim number", cl.claim_id), ("Policy number", cl.policy_number))
    p.section("Policyholder / claimant")
    p.kv("Name", cl.claimant.name)
    p.kv("Address", cl.claimant.address)
    p.kv2(("Phone", cl.claimant.phone), ("Email", cl.claimant.email))
    p.section("Insured vehicle")
    p.kv2(("Vehicle", cl.vehicle.label), ("Plate", cl.vehicle.plate))
    p.kv("VIN", cl.vehicle.vin)
    p.section("Loss details")
    p.kv2(("Date of loss", _d(cl.date_of_loss)), ("Time of loss", cl.time_of_loss))
    p.kv("Location", cl.loss_location)
    p.kv("Police notified", "Yes" if cl.police else "No")
    p.kv("Page", "1 of 2")
    page1 = RenderedPage(f"{cl.claim_id}/claim_form", "claim_form", p.bytes(), {
        "policy_number": cl.policy_number,
        "claimant_phone": cl.claimant.phone,
        "vehicle": cl.vehicle.label,
        "date_of_loss": _d(cl.date_of_loss),
        "loss_location": cl.loss_location,
    })

    p = Page()
    p.banner("Automobile Loss Notice - Description and Signature", INSURER, INSURER_ADDRESS)
    p.kv2(("Claim number", cl.claim_id), ("Policy number", cl.policy_number))
    p.section("Description of accident")
    p.para(
        f"On {_d(cl.date_of_loss)} at approximately {cl.time_of_loss}, I was driving my {cl.vehicle.label} "
        f"near {cl.loss_location} when I was {cl.loss_type}. The vehicle was damaged and "
        f"{'had to be towed from the scene' if cl.towing else 'was driven home after the accident'}."
    )
    p.section("Injuries")
    p.kv("Injuries reported", "Yes" if cl.injury else "No")
    if cl.injury:
        p.para("I felt neck and back pain after the collision and went for medical treatment.")
    p.section("Certification")
    p.para("I certify that the information on this notice is true and complete to the best of my knowledge. "
           "Any person who knowingly files a false claim may be guilty of insurance fraud.", size=8.5, leading=11)
    p.kv("Signature", cl.claimant.name)
    p.kv("Date signed", _d(cl.date_of_loss))
    p.kv("Page", "2 of 2")
    page2 = RenderedPage(f"{cl.claim_id}/claim_form", "claim_form", p.bytes(), {
        "injury_reported": "Yes" if cl.injury else "No",
    })
    return [page1, page2]


def police_report(cl: Claim) -> list[RenderedPage]:
    pr = cl.police
    if pr is None:
        return []
    p = Page()
    p.banner("Motor Vehicle Accident Report", pr.department.upper(), "Records Division")
    p.kv2(("Report number", pr.report_no), ("Date", _d(cl.date_of_loss)))
    p.kv2(("Time", cl.time_of_loss), ("Location", cl.loss_location))
    p.kv2(("Reporting officer", pr.officer), ("Badge", pr.badge))
    p.section("Vehicle 1")
    p.kv2(("Driver", cl.claimant.name), ("Vehicle", cl.vehicle.label))
    p.kv2(("Plate", cl.vehicle.plate), ("Phone", cl.claimant.phone))
    p.section("Vehicle 2")
    p.kv2(("Driver", pr.other_driver.name), ("Vehicle", pr.other_vehicle.label))
    p.kv2(("Plate", pr.other_vehicle.plate), ("Address", pr.other_driver.city + ", " + pr.other_driver.state))
    p.section("Officer narrative")
    p.para(f"Officer responded to a two-vehicle collision at {cl.loss_location}. Based on driver statements and "
           f"physical evidence, Vehicle 1 was {cl.loss_type}. No hazardous conditions were noted.")
    p.section("Findings")
    p.kv("At-fault driver", pr.at_fault)
    p.kv("Citation issued", pr.citation)
    p.kv("Injuries", "Possible injury, Vehicle 1" if cl.injury else "None reported")
    return [RenderedPage(f"{cl.claim_id}/police_report", "police_report", p.bytes(), {
        "police_report_number": pr.report_no,
        "at_fault_driver": pr.at_fault,
        "date_of_loss": _d(cl.date_of_loss),
    })]


def repair_estimates(cl: Claim) -> list[RenderedPage]:
    out = []
    for i, e in enumerate(cl.estimates):
        p = Page()
        p.banner("Repair Estimate", e.shop, f"{e.address} | {e.phone}")
        p.kv2(("Estimate number", e.estimate_no), ("Estimate date", _d(e.date)))
        p.kv2(("Customer", cl.claimant.name), ("Insurer claim", cl.claim_id))
        p.kv2(("Vehicle", cl.vehicle.label), ("VIN", cl.vehicle.vin))
        p.section("Parts and operations")
        p.table(["#", "Description", "Amount"],
                [[str(n), d, money(a)] for n, (d, a) in enumerate(e.lines, 1)],
                [30, 330, W - 2 * MARGIN - 360])
        p.table(["", "Labor", "Amount"],
                [["", f"Body labor {e.labor_hours} hrs @ {money(e.labor_rate)}/hr", money(e.labor)]],
                [30, 330, W - 2 * MARGIN - 360])
        p.total_line("Subtotal", money(e.subtotal))
        p.total_line("Sales tax (6.35%)", money(e.tax))
        p.total_line("Estimate total", money(e.total), bold=True)
        p.y -= 10
        p.para("This estimate is based on visible damage. Hidden damage found during repair will be submitted "
               "as a supplement for insurer approval.", size=8.5, leading=11)
        out.append(RenderedPage(f"{cl.claim_id}/estimate_{i + 1}", "repair_estimate", p.bytes(), {
            f"estimate_total:{i}": money(e.total),
            f"estimate_shop:{i}": e.shop,
        }))
    return out


def medical_bill(cl: Claim) -> list[RenderedPage]:
    m = cl.medical
    if m is None:
        return []
    p = Page()
    p.banner("Patient Statement / Itemized Bill", m.provider, m.address)
    p.kv2(("Patient", cl.claimant.name), ("Account number", m.account_no))
    p.kv2(("Date of service", _d(m.date_of_service)), ("Statement date", _d(cl.ack_date)))
    p.kv("Diagnosis (ICD-10)", f"{m.icd[0]} - {m.icd[1]}")
    p.section("Services")
    p.table(["CPT", "Description", "Charge"],
            [[c, d, money(a)] for c, d, a in m.lines],
            [60, 300, W - 2 * MARGIN - 360])
    p.total_line("Total charges", money(m.total), bold=True)
    p.total_line("Payments / adjustments", money(0))
    p.total_line("Balance due", money(m.total), bold=True)
    p.y -= 10
    p.para(f"Auto accident related. Please bill auto carrier: {INSURER}, claim {cl.claim_id}.", size=9)
    return [RenderedPage(f"{cl.claim_id}/medical_bill", "medical_bill", p.bytes(), {
        "medical_total": money(m.total),
        "diagnosis": m.icd[1],
    })]


def correspondence(cl: Claim, rng: random.Random) -> list[RenderedPage]:
    out = []

    # Adjuster acknowledgement email.
    p = Page()
    c = p.c
    c.setFont("Helvetica", 9.5)
    y = H - MARGIN
    for label, val in [("From", f"{cl.adjuster} <{cl.adjuster_email}>"),
                       ("To", f"{cl.claimant.name} <{cl.claimant.email}>"),
                       ("Date", _d(cl.ack_date)),
                       ("Subject", f"Claim {cl.claim_id} received - next steps")]:
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(MARGIN, y, f"{label}:")
        c.setFont("Helvetica", 9.5)
        c.drawString(MARGIN + 60, y, val)
        y -= 15
    c.line(MARGIN, y, W - MARGIN, y)
    p.y = y - 24
    first = cl.claimant.name.split()[0]
    p.para(f"Dear {first},")
    p.para(f"Thank you for reporting your loss of {_d(cl.date_of_loss)}. I am the adjuster assigned to claim "
           f"{cl.claim_id}. Please send the repair estimate, photos of the damage, and "
           f"{'any medical bills related to the accident' if cl.injury else 'the police report if available'}.")
    p.para("You can reply to this email or upload documents in the claims portal. I will contact the repair "
           "shop to schedule an inspection.")
    p.para(f"Regards,\n{cl.adjuster}\nClaims Adjuster, {INSURER}")
    out.append(RenderedPage(f"{cl.claim_id}/ack_email", "correspondence", p.bytes(), {
        "ack_date": _d(cl.ack_date),
        "adjuster": cl.adjuster,
    }))

    # Claimant letter (most packets).
    if rng.random() < 0.7:
        p = Page()
        p.y = H - MARGIN - 10
        for line in [cl.claimant.name, cl.claimant.street,
                     f"{cl.claimant.city}, {cl.claimant.state} {cl.claimant.zip}", "", _d(cl.ack_date), "",
                     INSURER, "Claims Department", INSURER_ADDRESS, ""]:
            p.c.setFont("Helvetica", 10)
            p.c.drawString(MARGIN, p.y, line)
            p.y -= 14
        p.para(f"Re: Claim {cl.claim_id}, policy {cl.policy_number}")
        p.para("To whom it may concern,")
        p.para(f"I am writing about the accident on {_d(cl.date_of_loss)}. My {cl.vehicle.label} has been out of "
               f"service since the accident and I need it for work. Please let me know if a rental car is covered "
               f"under my policy and when I can expect the repair to be approved.")
        p.para("I have attached the documents I have so far. Thank you for your help.")
        p.para(f"Sincerely,\n{cl.claimant.name}\n{cl.claimant.phone}")
        out.append(RenderedPage(f"{cl.claim_id}/claimant_letter", "correspondence", p.bytes(), {}))
    return out


def towing_receipt(cl: Claim) -> list[RenderedPage]:
    t = cl.towing
    if t is None:
        return []
    p = Page()
    p.banner("Towing Invoice", t.company, "24/7 Roadside | Licensed and insured")
    p.kv2(("Invoice number", t.invoice_no), ("Date", _d(cl.date_of_loss)))
    p.kv2(("Customer", cl.claimant.name), ("Vehicle", cl.vehicle.label))
    p.kv("Pickup", cl.loss_location)
    p.kv("Drop-off", cl.estimates[0].shop)
    p.section("Charges")
    hook = 95.0
    mileage = round(t.amount - hook, 2)
    p.table(["", "Item", "Amount"],
            [["", "Hook-up fee", money(hook)], ["", f"Mileage ({t.miles} mi)", money(mileage)]],
            [30, 330, W - 2 * MARGIN - 360])
    p.total_line("Amount paid", money(t.amount), bold=True)
    return [RenderedPage(f"{cl.claim_id}/towing", "other", p.bytes(), {"tow_amount": money(t.amount)})]


VIEWS = ["front bumper", "driver side door", "passenger side", "rear quarter panel",
         "headlight", "hood", "wide shot of the scene", "rear bumper"]
CAR_COLORS = [(170, 20, 30), (25, 40, 90), (200, 200, 205), (30, 30, 30), (240, 240, 240), (60, 90, 60)]


def _photo(rng: random.Random) -> bytes:
    """Draw a simple synthetic damage photo: car shape, damage patch, sensor noise, blur."""
    w, h = 900, 600
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    sky = (rng.randint(140, 200), rng.randint(170, 215), rng.randint(200, 245))
    ground = (rng.randint(70, 110),) * 3
    horizon = rng.randint(260, 340)
    d.rectangle([0, 0, w, horizon], fill=sky)
    d.rectangle([0, horizon, w, h], fill=ground)

    color = rng.choice(CAR_COLORS)
    x0, y0 = rng.randint(60, 200), rng.randint(250, 330)
    bw, bh = rng.randint(480, 640), rng.randint(120, 170)
    d.polygon([(x0 + bw * 0.22, y0), (x0 + bw * 0.35, y0 - bh * 0.7),
               (x0 + bw * 0.72, y0 - bh * 0.7), (x0 + bw * 0.85, y0)], fill=color)
    d.polygon([(x0 + bw * 0.27, y0 - 4), (x0 + bw * 0.37, y0 - bh * 0.6),
               (x0 + bw * 0.70, y0 - bh * 0.6), (x0 + bw * 0.80, y0 - 4)], fill=(120, 150, 175))
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=28, fill=color)
    for wx in (x0 + bw * 0.2, x0 + bw * 0.8):
        r = bh * 0.42
        d.ellipse([wx - r, y0 + bh - r * 0.9, wx + r, y0 + bh + r * 1.1], fill=(20, 20, 20))
        d.ellipse([wx - r * 0.45, y0 + bh - r * 0.35, wx + r * 0.45, y0 + bh + r * 0.55], fill=(150, 150, 150))

    cx, cy = x0 + rng.uniform(0.05, 0.95) * bw, y0 + rng.uniform(0.2, 0.8) * bh
    pts = [(cx + rng.uniform(-70, 70), cy + rng.uniform(-45, 45)) for _ in range(9)]
    d.polygon(pts, fill=tuple(max(0, v - 70) for v in color))
    for _ in range(rng.randint(4, 10)):
        sx, sy = cx + rng.uniform(-90, 90), cy + rng.uniform(-50, 50)
        d.line([sx, sy, sx + rng.uniform(-60, 60), sy + rng.uniform(-15, 15)], fill=(210, 210, 210), width=2)

    if rng.random() < 0.5:  # close-up shot
        zx, zy = int(max(0, cx - 260)), int(max(0, cy - 170))
        img = img.crop((zx, zy, min(w, zx + 520), min(h, zy + 347))).resize((w, h))
    noise = Image.effect_noise((w, h), rng.uniform(15, 35)).convert("RGB")
    img = Image.blend(img, noise, 0.12).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(55, 85))
    return buf.getvalue()


def photos(cl: Claim, rng: random.Random, pool: list[Path] | None = None) -> list[RenderedPage]:
    """Claimant damage photos: real photos from `pool` when given, else drawn ones."""
    n = rng.randint(2, 6)
    picks = rng.sample(pool, k=min(n, len(pool))) if pool else [None] * n
    out = []
    for k, src in enumerate(picks, 1):
        p = Page()
        if src is not None:  # real photos are square
            iw = ih = W - 2 * MARGIN - 80
            img = ImageReader(str(src))
            caption = f"IMG_{rng.randint(1000, 9999)}.jpg - damage photo {k} of {len(picks)} - uploaded {_d(cl.ack_date)}"
        else:
            iw = W - 2 * MARGIN
            ih = iw * 2 / 3
            img = ImageReader(io.BytesIO(_photo(rng)))
            caption = f"IMG_{rng.randint(1000, 9999)}.jpg - {rng.choice(VIEWS)} - uploaded {_d(cl.ack_date)}"
        p.c.drawImage(img, MARGIN, H - MARGIN - ih, iw, ih)
        p.c.setFont("Helvetica", 9)
        p.c.drawString(MARGIN, H - MARGIN - ih - 16, caption)
        out.append(RenderedPage(f"{cl.claim_id}/photo_{k}", "photo", p.bytes(), {},
                                source=src.name if src is not None else "drawn"))
    return out


def render_packet(cl: Claim, rng: random.Random, photo_pool: list[Path] | None = None) -> list[RenderedPage]:
    """All pages of the packet, in document order (shuffling happens at assembly)."""
    return (claim_form(cl) + police_report(cl) + repair_estimates(cl) + medical_bill(cl)
            + correspondence(cl, rng) + towing_receipt(cl) + photos(cl, rng, photo_pool))
