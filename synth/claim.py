"""Build the facts of one synthetic auto claim.

Every document in a packet (claim form, police report, estimates, bills,
letters, photos) is rendered from one `Claim`, so the facts agree across
pages and the ground truth is known exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from faker import Faker

INSURER = "Harborline Mutual Insurance"
INSURER_ADDRESS = "400 Commerce Way, Hartford, CT 06103"
DATE_FMT = "%m/%d/%Y"

VEHICLES = [
    ("Toyota", "Camry"), ("Honda", "Civic"), ("Ford", "F-150"), ("Chevrolet", "Malibu"),
    ("Nissan", "Altima"), ("Hyundai", "Elantra"), ("Subaru", "Outback"), ("Tesla", "Model 3"),
    ("Jeep", "Grand Cherokee"), ("Kia", "Sorento"), ("Mazda", "CX-5"), ("BMW", "330i"),
]

# (description, low, high) parts/labor line items for repair estimates.
REPAIR_ITEMS = [
    ("Front bumper cover - replace", 380, 960),
    ("Rear bumper cover - replace", 360, 900),
    ("Headlamp assembly LH - replace", 240, 1150),
    ("Headlamp assembly RH - replace", 240, 1150),
    ("Hood panel - repair", 180, 620),
    ("Front fender LH - repair", 220, 680),
    ("Front door shell RH - replace", 540, 1400),
    ("Quarter panel - repair", 300, 980),
    ("Radiator support - replace", 260, 740),
    ("Grille assembly - replace", 120, 480),
    ("Refinish and clear coat", 280, 860),
    ("Wheel alignment", 90, 180),
    ("Side mirror assembly - replace", 140, 520),
    ("Hazardous waste disposal", 5, 25),
]

# (CPT code, description, low, high) medical bill lines.
CPT_LINES = [
    ("99283", "Emergency dept visit, moderate severity", 380, 720),
    ("72040", "X-ray cervical spine, 2-3 views", 120, 260),
    ("99213", "Office visit, established patient", 110, 210),
    ("97110", "Therapeutic exercise, 15 min", 60, 140),
    ("97140", "Manual therapy, 15 min", 55, 120),
    ("73110", "X-ray wrist, 3 views", 95, 210),
]

ICD10 = [
    ("S13.4XXA", "Sprain of ligaments of cervical spine"),
    ("S16.1XXA", "Strain of muscle and tendon at neck level"),
    ("S39.012A", "Strain of muscle of lower back"),
    ("S60.511A", "Abrasion of right hand"),
]

LOSS_TYPES = [
    "rear-ended at a stop light",
    "sideswiped while changing lanes",
    "struck at an intersection by a vehicle running a red light",
    "hit while backing out of a parking space",
    "collided with a vehicle that failed to yield",
]


def money(x: float) -> str:
    return f"${x:,.2f}"


@dataclass
class Person:
    name: str
    phone: str
    email: str
    street: str
    city: str
    state: str
    zip: str

    @property
    def address(self) -> str:
        return f"{self.street}, {self.city}, {self.state} {self.zip}"


@dataclass
class Vehicle:
    year: int
    make: str
    model: str
    vin: str
    plate: str

    @property
    def label(self) -> str:
        return f"{self.year} {self.make} {self.model}"


@dataclass
class Estimate:
    shop: str
    address: str
    phone: str
    estimate_no: str
    date: date
    lines: list[tuple[str, float]]
    labor_hours: float
    labor_rate: float
    vehicle_label: str | None = None   # set only by an injected inconsistency

    @property
    def labor(self) -> float:
        return round(self.labor_hours * self.labor_rate, 2)

    @property
    def subtotal(self) -> float:
        return round(sum(a for _, a in self.lines) + self.labor, 2)

    @property
    def tax(self) -> float:
        return round(self.subtotal * 0.0635, 2)

    @property
    def total(self) -> float:
        return round(self.subtotal + self.tax, 2)


@dataclass
class MedicalBill:
    provider: str
    address: str
    account_no: str
    date_of_service: date
    lines: list[tuple[str, str, float]]
    icd: tuple[str, str]

    @property
    def total(self) -> float:
        return round(sum(a for _, _, a in self.lines), 2)


@dataclass
class PoliceReport:
    department: str
    report_no: str
    officer: str
    badge: str
    other_driver: Person
    other_vehicle: Vehicle
    at_fault: str
    citation: str
    report_date: date | None = None    # set only by an injected inconsistency


@dataclass
class Towing:
    company: str
    invoice_no: str
    miles: int
    amount: float


@dataclass
class Claim:
    claim_id: str
    policy_number: str
    claimant: Person
    vehicle: Vehicle
    date_of_loss: date
    time_of_loss: str
    loss_street: str
    loss_city: str
    loss_state: str
    loss_type: str
    injury: bool
    adjuster: str
    adjuster_email: str
    ack_date: date
    police: PoliceReport | None
    estimates: list[Estimate] = field(default_factory=list)
    medical: MedicalBill | None = None
    towing: Towing | None = None
    form_injury: str | None = None     # what the loss notice says, if it disagrees with reality
    anomalies: list[str] = field(default_factory=list)

    @property
    def injury_on_form(self) -> str:
        return self.form_injury or ("Yes" if self.injury else "No")

    @property
    def police_date(self) -> date | None:
        return None if self.police is None else (self.police.report_date or self.date_of_loss)

    @property
    def loss_location(self) -> str:
        return f"{self.loss_street}, {self.loss_city}, {self.loss_state}"

    @property
    def lowest_estimate(self) -> Estimate:
        return min(self.estimates, key=lambda e: e.total)


def _person(fake: Faker, rng: random.Random) -> Person:
    first, last = fake.first_name(), fake.last_name()
    return Person(
        name=f"{first} {last}",
        phone=f"({rng.randint(201, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
        email=f"{first.lower()}.{last.lower()}@{fake.free_email_domain()}",
        street=fake.street_address(),
        city=fake.city(),
        state=fake.state_abbr(include_territories=False),
        zip=fake.zipcode(),
    )


def _vehicle(fake: Faker, rng: random.Random) -> Vehicle:
    make, model = rng.choice(VEHICLES)
    return Vehicle(
        year=rng.randint(2012, 2026),
        make=make,
        model=model,
        vin=fake.vin(),
        plate=fake.license_plate().replace(" ", ""),
    )


def _estimate(fake: Faker, rng: random.Random, loss: date, base_items: list) -> Estimate:
    # Shops quote the same damage, with their own prices and sometimes an extra line.
    items = list(base_items)
    if rng.random() < 0.4:
        extra = rng.choice([i for i in REPAIR_ITEMS if i not in items])
        items.append(extra)
    lines = [(desc, round(rng.uniform(lo, hi), 2)) for desc, lo, hi in items]
    return Estimate(
        shop=f"{fake.last_name()} {rng.choice(['Auto Body', 'Collision Center', 'Body & Paint', 'Collision Repair'])}",
        address=fake.address().replace("\n", ", "),
        phone=f"({rng.randint(201, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
        estimate_no=f"EST-{rng.randint(10000, 99999)}",
        date=loss + timedelta(days=rng.randint(2, 9)),
        lines=lines,
        labor_hours=round(rng.uniform(4, 22), 1),
        labor_rate=float(rng.choice([58, 62, 65, 68, 72, 75])),
    )


def build_claim(index: int, rng: random.Random, fake: Faker) -> Claim:
    claimant = _person(fake, rng)
    loss = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    injury = rng.random() < 0.45
    has_police = rng.random() < 0.75

    police = None
    if has_police:
        other = _person(fake, rng)
        police = PoliceReport(
            department=f"{claimant.city} Police Department",
            report_no=f"{loss.year}-{rng.randint(100000, 999999)}",
            officer=f"Ofc. {fake.first_name()} {fake.last_name()}",
            badge=str(rng.randint(1000, 9999)),
            other_driver=other,
            other_vehicle=_vehicle(fake, rng),
            at_fault=other.name if rng.random() < 0.8 else claimant.name,
            citation=rng.choice(["VTL 1111(d) failure to obey traffic signal",
                                 "VTL 1129(a) following too closely",
                                 "VTL 1128(a) unsafe lane change",
                                 "VTL 1143 failure to yield right of way"]),
        )

    base_items = rng.sample(REPAIR_ITEMS, k=rng.randint(3, 6))
    n_estimates = 2 if rng.random() < 0.7 else 1
    estimates = [_estimate(fake, rng, loss, base_items) for _ in range(n_estimates)]

    medical = None
    if injury:
        medical = MedicalBill(
            provider=f"{fake.last_name()} {rng.choice(['Urgent Care', 'Orthopedics', 'Medical Group', 'Physical Therapy'])}",
            address=fake.address().replace("\n", ", "),
            account_no=f"ACCT-{rng.randint(100000, 999999)}",
            date_of_service=loss + timedelta(days=rng.randint(0, 3)),
            lines=[(c, d, round(rng.uniform(lo, hi), 2)) for c, d, lo, hi in rng.sample(CPT_LINES, k=rng.randint(2, 4))],
            icd=rng.choice(ICD10),
        )

    towing = None
    if rng.random() < 0.5:
        miles = rng.randint(3, 40)
        towing = Towing(
            company=f"{fake.last_name()} Towing & Recovery",
            invoice_no=f"T-{rng.randint(1000, 9999)}",
            miles=miles,
            amount=round(95 + miles * rng.uniform(3.5, 6.0), 2),
        )

    adj_first, adj_last = fake.first_name(), fake.last_name()
    return Claim(
        claim_id=f"CLM-2026-{index:05d}",
        policy_number=f"HMI-AU-{rng.randint(1000000, 9999999)}",
        claimant=claimant,
        vehicle=_vehicle(fake, rng),
        date_of_loss=loss,
        time_of_loss=f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {rng.choice(['AM', 'PM'])}",
        loss_street=f"{fake.street_name()} and {fake.street_name()}",
        loss_city=claimant.city,
        loss_state=claimant.state,
        loss_type=rng.choice(LOSS_TYPES),
        injury=injury,
        adjuster=f"{adj_first} {adj_last}",
        adjuster_email=f"{adj_first[0].lower()}{adj_last.lower()}@harborlinemutual.com",
        ack_date=loss + timedelta(days=rng.randint(1, 5)),
        police=police,
        estimates=estimates,
        medical=medical,
        towing=towing,
    )


ANOMALY_TYPES = ["police_date_mismatch", "injury_mismatch", "vehicle_mismatch", "treatment_before_loss"]


def inject_anomalies(cl: Claim, rng: random.Random, rate: float = 0.35) -> Claim:
    """Plant at most one realistic inconsistency in ~35% of claims, for deep-question tests.

    Uses its own `rng` so claims without an anomaly render exactly as before.
      police_date_mismatch   police report date differs from the loss notice by 1-3 days
      injury_mismatch        loss notice says "No" injuries, but a medical bill is in the packet
      vehicle_mismatch       one repair estimate lists a different vehicle than the loss notice
      treatment_before_loss  medical date of service is BEFORE the accident (possible fraud flag)
    """
    if rng.random() >= rate:
        return cl
    options = []
    if cl.police:
        options.append("police_date_mismatch")
    if cl.medical:
        options += ["injury_mismatch", "treatment_before_loss"]
    if cl.estimates:
        options.append("vehicle_mismatch")
    kind = rng.choice(options)
    if kind == "police_date_mismatch":
        cl.police.report_date = cl.date_of_loss + timedelta(days=rng.choice([-3, -2, -1, 1, 2, 3]))
    elif kind == "injury_mismatch":
        cl.form_injury = "No"
    elif kind == "treatment_before_loss":
        cl.medical.date_of_service = cl.date_of_loss - timedelta(days=rng.randint(1, 4))
    elif kind == "vehicle_mismatch":
        make, model = rng.choice([v for v in VEHICLES if v != (cl.vehicle.make, cl.vehicle.model)])
        i = rng.randrange(len(cl.estimates))
        cl.estimates[i].vehicle_label = f"{cl.vehicle.year} {make} {model}"
    cl.anomalies.append(kind)
    return cl
