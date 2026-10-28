#!/usr/bin/env python3
"""Deterministic generator for a FICTIONAL Obsidian-style vault plus a golden question set.

Everything here is invented: the studio, the clients, the people, the invoice numbers.
It exists to test a retrieval kit that indexes a stranger's markdown notes, so retrieval
has to be *measurable*: a small set of hand-written "plant" notes carry unique, checkable
facts, and the golden questions point at them. A large, seeded pile of plausible filler
notes (journal days, meetings, project notes) surrounds the plants. The filler is
topically similar to the plants (lots of "budget", "invoice", "permit") but never
contains the plants' specific facts.

Usage
    python make_vault.py --notes 1000 --seed 7 --out /some/dir
    python make_vault.py --self-check --notes 100

Output (byte-identical for identical args):
    OUT/vault/          the notes (+ .obsidian/, .trash/, attachments/)
    OUT/golden.json     {"version":1,"cases":[...]}
    OUT/nearmiss.json   {"version":1,"cases":[...]}: prompts that reuse the vault's vocabulary
                        (permit, invoice, staging API, sourdough...) but that the vault cannot
                        answer, so a good retriever should score them as "nothing relevant"

Sizes. Below N=100 the vault holds the original 27 plant notes and a small golden set
(see SMALL_N_CAPS). From N=100 up it also holds 43 extra plant notes (PLANTS_EXT, each with
2 to 3 independent facts), which lifts golden.json to 106 answerable cases plus 5 no_answer.
The original 30 cases and the original 27 plants never change, so earlier measurements stay
comparable.

A paraphrase case is checked mechanically by --self-check: tokenize the question and the gold
note (lowercase words of 3+ letters), drop stopwords and any word found in more than 3% of the
vault's notes, then compare by 5-letter prefix; the question must share zero tokens with the
gold note. The near-miss prompts are checked to contain no number or identifier (INV-1234,
ports, codes) that occurs in the vault.

Python 3.11+, standard library only. Mtimes are set relative to a fixed anchor date
(2026-10-26), never relative to "now".
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

ANCHOR = datetime(2026, 10, 26, tzinfo=timezone.utc)  # "today" inside the fictional vault
HORIZON_DAYS = 425  # about 14 months of mtime spread
MIN_NOTES = 30

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --------------------------------------------------------------------------------------
# Plant notes: fixed content, present at every N. Golden cases point at these.
# Each entry: (vault-relative path, "YYYY-MM-DD HH:MM" mtime, text)
# --------------------------------------------------------------------------------------

PLANTS: list[tuple[str, str, str]] = []


def plant(path: str, when: str, text: str) -> None:
    PLANTS.append((path, when, text.strip("\n") + "\n"))


# ---- direct (8) ----------------------------------------------------------------------

plant("Projects/Harbor/permit-timeline.md", "2026-02-12 10:20", """
---
tags: [harbor, permits, tidewater]
created: 2026-01-28
status: waiting
---
# Harbor sign permit timeline

Tidewater Bakery wants the new blade sign and the storefront vinyl up before the spring tourist rush, so the city sign permit is the long pole on this job.

## Filing
- Application SP-2291 was filed with the city planning desk on February 9.
- The planner assigned to us is Dale Whitcombe. He says review takes six weeks if nothing bounces.
- Filing fee was $215, receipt noted in [[invoice-log]].

## Expected dates
Approval is expected by March 23 as long as the structural drawing for the bracket passes the first read. If it bounces, add two weeks and tell Ruth at the bakery right away.

> [!note]
> Do not order the fabricated blade sign until the permit is stamped. The sign shop wants a 50% deposit and does not refund it.

## Next
- [ ] Email Dale to confirm the bracket drawing was received
- [ ] Ask Tidewater whether the awning company needs its own permit

#harbor #permits
""")

plant("Projects/Juniper/hosting-setup.md", "2026-04-02 15:05", """
---
tags: [juniper, hosting, web]
created: 2026-03-30
aliases: [Juniper hosting]
client: Juniper Veterinary
---
# Juniper Veterinary hosting setup

I compared three hosts for the clinic site. The pick is **Fernhost Standard at $27.40/month** because it includes daily backups and a staging site, which Odalys (the clinic manager) asked for after their last site vanished with no restore point.

| Host | Plan | Monthly | Staging | Daily backups |
|---|---|---|---|---|
| Fernhost | Standard | $27.40 | yes | yes |
| Cloudpine | Basic | $19.00 | no | weekly |
| Brindle Cloud | Pro | $41.25 | yes | yes |

Billing is annual, renewing in April. Domain stays at the registrar the clinic already uses; I only need the DNS login. See [[Projects/Juniper/overview|the Juniper overview]] for scope.
""")

plant("Projects/Kestrel/api-notes.md", "2026-05-20 09:40", """
---
tags: [kestrel, api, dev]
created: 2026-05-02
---
# Kestrel API notes

The staging API for the Kestrel app listens on port 8421 and is only reachable over the VPN. Production is plain 443. Auth is a bearer token kept in the shared password vault under "Kestrel".

| Endpoint | Method | Notes |
|---|---|---|
| /v2/shipments | GET | paginated, 50 per page |
| /v2/shipments/{id}/events | GET | newest first |
| /v2/devices | POST | registers a handheld scanner |

```bash
curl -H "Authorization: Bearer $TOKEN" https://staging.kestrel.example:8421/v2/shipments?page=1
```

Rate limit is generous on staging but the mobile team keeps tripping it with polling. Suggest websockets in the next planning call.
""")

plant("Meetings/2026-03-12 Marlow packaging kickoff.md", "2026-03-12 17:45", """
---
tags: [meeting, marlow, packaging]
created: 2026-03-12
---
# Marlow packaging kickoff

**Attendees:** Maren, Beatrix Ng (Marlow Tea), Jonas from the print side

## Notes
- Three carton sizes for the first range: 20-bag, 50-bag and the gift tin sleeve.
- Beatrix agreed to a cap of 4 dieline revisions; anything past that is billed hourly.
- Photography is out of scope, Marlow already has the leaf shots.

## Budget
Design fee is fixed. Print costs are Marlow's own account with their supplier.

## Action items
- [ ] Maren to send the dieline templates
- [ ] Beatrix to confirm the legal copy for the back panel
""")

plant("Reference/Client invoice SOP.md", "2026-01-20 11:00", """
---
tags: [sop, invoicing, studio]
created: 2025-12-18
---
# Client invoice SOP

1. Invoices go out on the last Thursday of the month, or at milestone sign-off for fixed-price jobs.
2. Use the invoice template in the finance folder and number it in sequence.
3. Terms are net 30 unless the client file says otherwise.
4. I add a 1.5% late fee per month on anything overdue, stated on the invoice itself so nobody is surprised.
5. Chase on day 3 after the due date with a friendly note, then again at day 10 by phone.

> [!tip]
> Put the project name in the invoice subject line. Accounts-payable teams search for it.
""")

plant("Reading/Atomic Interfaces - notes.md", "2026-06-30 20:15", """
---
tags: [reading, ui, typography]
author: Soren Valtonen
rating: 4
---
# Atomic Interfaces (Valtonen)

Short book, good chapters on type. Notes while reading:

## Chapter 3: Reading width
Valtonen says body text should sit at roughly 62 characters per line. Anything past 75 and eyes lose their place on the return sweep; anything under 40 chops the sentences up.

## Chapter 5: Component naming
Name components by role, not look. "Primary button" survives a rebrand, "blue button" does not.

## Takeaways
- Check line length on the real content, not lorem ipsum.
- Rename the old Figma library layers this week.
""")

plant("Reference/Sourdough starter schedule.md", "2026-07-09 08:30", """
---
tags: [recipe, bread]
---
# Sourdough starter schedule

Feed the starter 1:2:2 (starter, flour, water) the evening before baking and keep it on the counter.

## Bake day
1. Mix the dough in the morning, four folds across the first two hours.
2. Shape, then put the loaf in the fridge to proof for 19 hours. Longer than 24 and it turns sour and slack.
3. Bake straight from the fridge in a preheated dutch oven, 245C lid on for 20 minutes, lid off for 22.

Cool for an hour before slicing, no exceptions.
""")

plant("Ideas/Studio newsletter.md", "2026-09-02 21:10", """
---
tags: [idea, marketing]
status: maybe
---
# Studio newsletter

Monthly, not weekly, so it actually ships. One teardown of a piece of work from the studio, one tool I'm using, one link I'm embarrassed to have missed.

Working name: **Overset Weekly**. It's a joke about text that overflows its box, and I know it says weekly while being monthly. I'll decide whether that bugs me before the first issue.

- [ ] Draft issue zero
- [ ] Pick the email tool
""")

# ---- paraphrase (8) -------------------------------------------------------------------

plant("Projects/House/plumbing-schedule.md", "2026-10-08 18:20", """
---
tags: [house, renovation]
created: 2026-10-01
---
# Kitchen plumbing schedule

Gus from Redwater Pipe & Heat is booked for the 14th of November to replace the kitchen supply line and the leaky shutoff valve. Deposit of $380 went over by e-transfer today.

Clear everything out from under the sink the night before. He needs the whole cabinet empty and the water off at the main by 8.

Still to decide: whether to redo the faucet at the same time. Quartz offcut for the counter is a separate job.
""")

plant("Journal/2026-09-18.md", "2026-09-18 22:05", """
Laptop battery swelled up and lifted the trackpad off the case. Took it to the repair counter after lunch; they quoted a 6-day turnaround and $149 for a new cell. Working off the old iPad until it's back, which is fine for email and miserable for design.

Cooked lentil soup, ate it with the good bread. Early night.

#journal
""")

plant("Clients/Halvorsen & Daughters (furniture).md", "2026-02-03 14:30", """
---
tags: [client, furniture]
aliases: [Halvorsen]
created: 2025-10-22
---
# Halvorsen & Daughters (furniture)

Third-generation shop that builds custom cabinets and dining tables. Owner is Ingrid Halvorsen, and the shop floor is run by her sister Karin.

## How to reach them
Ingrid answers on Signal within minutes; email sits for days. Karin doesn't do messages at all, call the shop landline before 9.

## Billing
Net-45, always chases late. Pay by cheque, which still surprises me.

## Work so far
Catalogue layout (done), a new stamp mark for the underside of pieces (in progress).
""")

plant("Reference/Quarterly tax set-aside.md", "2026-01-06 09:00", """
---
tags: [finance, taxes]
---
# Quarterly tax set-aside

Move 28% of every client payment into the tax savings account the day it lands. Don't touch it for anything else, including emergencies; that's what the other account is for.

Estimated payments go out in January, April, June and September. Put reminders on the calendar a week early so the transfer clears.

Keep receipts in the monthly folder, not in the inbox.
""")

plant("Meetings/2026-05-07 Kestrel retro.md", "2026-05-07 16:50", """
---
tags: [meeting, kestrel, retro]
created: 2026-05-07
---
# Kestrel retro

## What went well
- Shipment list screen landed on time.
- Handheld scanner flow tested with real devices.

## What didn't
- Too many late scope changes in the last week.

## Decisions
Decision: dark mode is out of v1 because the contrast audit failed on the shipment status chips, and fixing them properly would slip the release by two weeks. Revisit after launch.

## Action items
- [ ] File the contrast issues as tickets
- [ ] Move the dark palette work to the backlog
""")

plant("Reading/The Quiet Grid - Brandvold.md", "2026-08-11 19:40", """
---
tags: [reading, layout, business]
author: Ola Brandvold
rating: 5
---
# The Quiet Grid (Brandvold)

Best business-of-design book I've read this year.

> "Empty space is a deliverable. Charge for it."

Brandvold's pitch: list whitespace as a line item on the quote, so clients stop asking you to fill it. The same logic applies to revision rounds and to the thinking time before any pixels move.

## To try
- Rewrite the quote template with a "composition and restraint" line
- Explain it to Bramble before the next proposal
""")

plant("Ideas/Podcast guest wishlist.md", "2026-07-30 12:25", """
---
tags: [idea, podcast]
---
# Podcast guest wishlist

If the podcast ever happens, these are the first people to ask:

- Priya Raman, runs Kiln & Co, a ceramics shop that sells glazes by the jar. Ask in January when the holiday rush is over.
- A letterpress printer, ideally someone with an opinion about paper weight
- Someone who has switched careers into design after thirty

No recording gear yet. Start with voice memos.
""")

plant("Reference/Backup routine.md", "2026-04-26 10:10", """
---
tags: [backup, sop]
---
# Backup routine

Time Machine goes to the blue drive every night while the laptop is docked.

Every first Sunday I clone the whole disk to the drive that lives at Mum's house, then bring the older one back. Offsite copy is the one that matters if the flat floods.

Check the disk health in the utility once a quarter. Replace drives at four years regardless.
""")

# ---- entity (4) ----------------------------------------------------------------------

plant("Projects/Harbor/invoice-log.md", "2026-03-30 13:35", """
---
tags: [harbor, invoices]
created: 2026-01-30
---
# Harbor invoice log

| Invoice | What | Amount | Status |
|---|---|---|---|
| INV-4452 | Permit fee reimbursement | $215 | paid |
| INV-4471 | Milestone 2: Tidewater window decals, vinyl and install | $3,260 | paid |
| INV-4488 | Blade sign artwork revisions | $640 | sent |

Reminder: the permit itself is tracked in [[permit-timeline]].
""")

plant("Projects/Kestrel/ticket-index.md", "2026-06-11 11:50", """
---
tags: [kestrel, tickets]
---
# Kestrel ticket index

- KST-301 shipment list skeleton loading state
- KST-318 offline sync crash on Android 14 when the scanner reconnects mid-upload
- KST-322 wrong timezone on event timestamps for the Auckland depot
- KST-340 accessibility labels on status chips

Triage happens on Mondays. Anything crash-related jumps the queue.
""")

plant("Reference/Suppliers.md", "2026-02-24 09:25", """
---
tags: [suppliers, reference]
---
# Suppliers

| Code | Supplier | Handles | Lead time | Contact |
|---|---|---|---|---|
| SUP-0788 | Nordhavn Sign Co | vinyl, sign fabrication | 8 business days | Pia |
| SUP-0912 | Lindqvist Print & Paper | cartons, labels, stationery boards | 11 business days | Hanne |
| SUP-1034 | Alder Frameworks | wooden display frames | 15 business days | Tomek |

Always get the quote in writing before the proof is approved.
""")

plant("Projects/Kestrel/release-notes.md", "2026-07-22 17:00", """
---
tags: [kestrel, release]
---
# Kestrel release notes

## 3.8.2-rc4
Adds the biometric login on supported handhelds, fixes the empty state on the shipments tab, and bumps the map SDK.

## 3.8.1
Hotfix for pagination skipping items on slow connections.

## 3.7.9
Last build before the navigation rewrite. Keep the tag around in case we need to roll back.
""")

# ---- multi_note extras (3) -------------------------------------------------------------

plant("Clients/Tidewater Bakery.md", "2026-04-15 10:45", """
---
tags: [client, tidewater]
aliases: [Tidewater]
---
# Tidewater Bakery

Owner Ruth Abara. Corner shop by the ferry landing, opens at 6. Pays quickly when the invoice reaches her personally, slowly when it goes to the general inbox.

## Billing notes
- INV-4502 covers the blade sign design, $1,980.
- Terms net 14.
- She prefers a paper copy with the signed proof.

See [[Projects/Harbor/permit-timeline|the permit timeline]] before promising install dates.
""")

plant("Projects/Marlow/print-spec.md", "2026-03-27 14:15", """
---
tags: [marlow, print]
---
# Marlow print spec

- Stock: 350gsm uncoated board, FSC certified
- Colours: four-colour plus one spot (tea leaf green)
- Finish: matte varnish on the front panel only
- Cartons are printed by SUP-0912, see the supplier sheet for the contact
- Proofs are approved by Beatrix and signed off before the plates are made

Delivery goes to Marlow's warehouse, not the studio.
""")

plant("Projects/House/payments-log.md", "2026-10-20 08:55", """
---
tags: [house, money]
---
# House payments log

| Date | Who | What | Amount |
|---|---|---|---|
| Sep 29 | Tile Barn | grout and sealer | $64.10 |
| Oct 12 | electrician | outlet moves | $210.00 |

Final balance owed to Redwater Pipe & Heat after the job is $1,240.75, due on completion.
""")

# ---- updated pairs (4) -----------------------------------------------------------------

plant("Clients/Kestrel Freight.md", "2025-11-04 10:00", """
---
tags: [client, kestrel]
aliases: [Kestrel]
---
# Kestrel Freight

Logistics company, mobile app for warehouse handhelds. Main contact is Amara Osei, product lead.

## Rates
Day rate: $640. Quoted in the original proposal and signed off by Amara. Invoiced monthly.

## Notes
Prefers a Monday check-in call. Legal needs two weeks for any contract change.
""")

plant("Projects/Kestrel/rate-change.md", "2026-08-18 09:30", """
---
tags: [kestrel, rates]
created: 2026-08-18
---
# Kestrel rate change

Update: from September the Kestrel day rate is $720. Amara agreed to it by email after the scope grew with the scanner work, and legal signed the amendment last week.

Existing signed estimates keep their old numbers until they expire. Anything new gets quoted at the new rate.
""")

plant("Meetings/2026-01-15 Studio ops sync.md", "2026-01-15 15:00", """
---
tags: [meeting, studio-ops]
created: 2026-01-15
---
# Studio ops sync

## Notes
- Quarterly review is held at the Cannery coworking space, second Tuesday of the quarter.
- Software subscriptions audit: cancel two seats we don't use.
- Keep the shared drive tidy: client folders only, no loose exports.

## Action items
- [ ] Book the Cannery room for April
""")

plant("Meetings/2026-08-20 Studio ops sync.md", "2026-08-20 15:30", """
---
tags: [meeting, studio-ops]
created: 2026-08-20
---
# Studio ops sync

## Notes
- Update: the quarterly review has moved to the Alder Street library meeting room, same second-Tuesday slot. The coworking space raised its hourly rate and the library room is free for members.
- Subscriptions: one seat dropped, saving a little each month.

## Action items
- [ ] Reserve the library room for the autumn review
""")

assert len(PLANTS) == 27, len(PLANTS)

# --------------------------------------------------------------------------------------
# Extension plants (v2): only written at N >= SMALL_N_LIMIT. Each carries 2 to 3
# independent facts in different sections or paragraphs, so the extra golden cases
# (see CASES_EXT) can each point at a different note. The 27 plants above are untouched.
# --------------------------------------------------------------------------------------

PLANTS_EXT: list[tuple[str, str, str]] = []


def plant_ext(path: str, when: str, text: str) -> None:
    PLANTS_EXT.append((path, when, text.strip("\n") + "\n"))


plant_ext("Clients/Fennick Optics.md", "2026-01-12 11:10", """
---
tags: [client, fennick]
aliases: [Fennick]
created: 2026-01-12
---
# Fennick Optics

Three-shop eyewear chain on the east side. Dorothea Lund runs brand and her brother Piet Lund handles the money side.

## Getting paid
Piet pays supplier bills in one batch on the 20th of the month, so an invoice has to be in his inbox a few days before that or it waits a whole cycle.

## Approvals
Dorothea will not sign off artwork on a screen. She tapes A3 printouts inside the shop window and looks at them from the pavement at different times of day.
""")

plant_ext("Meetings/2026-09-22 Fennick catch-up.md", "2026-09-22 16:30", """
---
tags: [meeting, fennick]
created: 2026-09-22
---
# Fennick catch-up

**Attendees:** Maren, Dorothea Lund, Piet Lund

## Notes
- Update: Piet has moved the Fennick payment run to the 5th of the month, starting with October's batch. Anything invoiced before the last Thursday makes that run.
- Dorothea wants the decals on the front glass of the Harbor Street shop in matte, not gloss, because the afternoon sun bounces off gloss and hides the lettering.
- Shop three reopens after the refit on November 3.

## Action items
- [ ] Send Dorothea the matte vinyl swatch
""")

plant_ext("Clients/Salt Row Distillery.md", "2026-05-18 10:00", """
---
tags: [client, salt-row]
created: 2026-04-30
---
# Salt Row Distillery

Small-batch gin and a rye that is still ageing. Owner is Anneke Voorhees, head distiller is Callum Frayne.

## Label runs
Bottling happens on the first Thursday of the month. Label files have to be at the printer the Monday before, no exceptions, because the line crew cannot wait for reprints.

## Compliance
The government warning block needs a minimum height of 2.5 mm. Anneke keeps a laminated template next to the line and checks every proof against it.

## People
Callum is red-green colour blind, so never rely on red against green to tell the gin variants apart. Use a shape or a number as well.

## Invoices
| Invoice | What | Amount | Status |
|---|---|---|---|
| INV-4920 | Gin label redesign | $2,750 | sent |
| INV-4933 | Rye label, first artwork round | $1,900 | draft |
""")

plant_ext("Clients/Greywell Dairy.md", "2026-03-20 09:45", """
---
tags: [client, greywell]
created: 2026-03-03
---
# Greywell Dairy

Milk, butter and a very good kefir. Contact is Imke Sandvik in marketing; payments go through their finance portal.

## Billing gotcha
Every invoice must carry the purchase order number in the subject line or the portal bounces it without telling anyone. Imke has to raise a new PO if the scope changes.

## Brand
The Greywell green is Pantone 7731 C. Never print it as a CMYK mix on uncoated stock, it goes muddy.

## Recent
INV-4941 for the milk carton redesign, $1,430, went out in March.
""")

plant_ext("Clients/Ostrander Architects.md", "2026-02-09 15:20", """
---
tags: [client, ostrander]
created: 2026-02-09
---
# Ostrander Architects

Mid-sized practice, mostly civic work. Tamsin Okoye signs everything off, and she hates being copied on threads. She wants one email a week, Fridays at 4, and nothing in between unless something is on fire.

## Job codes
| Code | Job |
|---|---|
| OA-26-031 | Courtyard housing brief |
| OA-26-044 | Chapel restoration signage |
| OA-26-052 | Pavilion wayfinding |
""")

plant_ext("Projects/Quillon/cover-series-notes.md", "2026-06-02 14:00", """
# Quillon cover series notes

Six covers for the Harrowgate trilogy reissue and the two follow-ups.

## Run size
Quillon's first press run is 2,400 copies, split evenly across the first three titles. A reprint only happens if the distributor orders more than 600 in a month.

## Finishing
The spine foil is done by Tern & Quill Bindery and adds 6 working days on top of the normal schedule. Book that slot before the proofs go out, not after.
""")

plant_ext("Projects/Quillon/proof-log.md", "2026-06-25 18:10", """
# Quillon proof log

PRF-118 cover A, rejected: the spine width was calculated from the wrong page count.
PRF-121 cover A, approved by Soraya on the Monday.
PRF-124 cover B, waiting on the foil sample.
""")

plant_ext("Projects/Bramble/frame-decals.md", "2026-07-14 11:30", """
---
tags: [bramble, decals]
created: 2026-07-14
---
# Bramble frame decals

The head-tube decal is 38 x 120 mm, white on matte black, with a 1 mm safety margin so the cutter never clips the edge.

## Supplier
Strand & Reeve Graphics (SUP-1177) cuts and weeds them. They ship in rolls of 200 and want artwork as outlined PDF only.
""")

plant_ext("Reference/Paper stocks.md", "2026-04-11 10:40", """
# Paper stocks

| Code | Stock | Weight | Used for |
|---|---|---|---|
| PS-210 | Hartwell Cream | 120gsm | letterheads |
| PS-233 | Ardent White | 300gsm | business cards |
| PS-248 | Birchline Kraft | 410gsm | mailer boxes |

Ask for a hand sample before quoting any of these; the cream drifts between mills.
""")

plant_ext("Projects/Juniper/ticket-index.md", "2026-08-03 16:05", """
# Juniper ticket index

- JNP-412 contact form sends duplicate emails when a visitor double-taps submit
- JNP-427 map pin still points at the old Kellner Street address
- JNP-431 alt text missing on the staff photos

Fix anything that touches the contact form before launch.
""")

plant_ext("Projects/Juniper/changelog.md", "2026-08-27 12:15", """
# Juniper site changelog

## v1.4.0
Adds SMS appointment reminders and the new booking form.

## v1.3.2
Fixes the footer map link.
""")

plant_ext("Projects/House/electrician-visit.md", "2026-09-24 19:20", """
---
tags: [house, electrical]
created: 2026-09-24
---
# Electrician visit

Wilf Pardoe came to look at the kitchen circuit. The new induction hob needs its own 40-amp breaker and a fatter cable run, which he can do in half a day.

His callout minimum is $95 even if the fix takes ten minutes, so batch the small jobs: the hallway socket and the dead porch light can go in the same visit.
""")

plant_ext("Journal/2026-10-03.md", "2026-10-03 21:00", """
Painted the hallway in Oyster Shell, two coats, a 2.5 litre tin from Dunmore Paints at $58. The cut-in along the ceiling took longer than the rolling.

The smoke alarm chirped at 3am. Swapped the 9V; the spare batteries live in the drawer under the kettle, which I always forget.

#journal
""")

plant_ext("Health/Physio plan.md", "2026-08-25 08:50", """
---
tags: [health, physio]
created: 2026-08-25
---
# Physio plan

Ines Calder, every Thursday at 8:15, for the left shoulder. Bring the resistance band.

## Home exercises
Band rows, three sets, holding each pull for 30 seconds. Stop if the elbow goes numb and tell Ines.
""")

plant_ext("Journal/2026-10-14.md", "2026-10-14 22:10", """
Update: Ines moved physio to Tuesdays at 7:40 from next week, because the Thursday slot went to the clinic's group class. Earlier start, so I have to be in bed by 10.

Bought running shoes in size 44 at Stride Lab, $139, after the gait test. Left foot rolls in a little.

#journal
""")

plant_ext("Health/Annual checkup.md", "2026-10-08 13:25", """
# Annual checkup

GP is Dr Aisling Rowe. Next appointment is January 27 at 9:20, fasting bloods the same morning so no breakfast.

## Supplements
Vitamin D, 2000 IU a day from October to March. Stop in April.
""")

plant_ext("Ideas/Autumn trip.md", "2026-09-06 20:30", """
---
tags: [idea, travel]
status: planned
---
# Autumn trip

Four nights in Tallinn at the end of October. Flight leaves at 07:25 on Friday, October 30; hotel is the Viru Lane Guesthouse at €92 a night.

Nothing booked for Saturday yet. Look for a sauna.
""")

plant_ext("Travel/Tallinn trip.md", "2026-10-15 09:05", """
---
tags: [travel, tallinn]
created: 2026-10-15
---
# Tallinn trip

Update: the airline retimed the outbound flight, it now leaves at 11:50 instead of the early slot. Means I can pack properly the morning of.

## Costs
The Tallinn Card for 48-hour entry to the museums and public transport is €55. Buy it online, the airport kiosk charges more.
""")

plant_ext("Travel/Mum visit.md", "2026-07-19 17:45", """
Train to Mum's for the weekend: the 9:12 from Platform 4, return ticket £34.50 if I book a week ahead.

The spare key is under the blue heron pot by the back door. Mum keeps forgetting to say, so I'm writing it down.
""")

plant_ext("Reference/Business accounts.md", "2026-05-27 10:15", """
# Business accounts

Operating account is at Larchmont Credit. The savings pot for the tax money earns 3.9% interest, paid monthly.

The company card statement closes on the 17th, so anything bought after that lands on next month's bill.
""")

plant_ext("Reference/Accountant.md", "2026-01-30 14:40", """
# Accountant

Nuala Brennan at Brennan & Pike. Bookkeeping help is $85 an hour, tax filing is a flat fee quoted each year.

Year-end meeting is booked for January 13 at 2pm at her office. Bring the mileage log and the card statements.
""")

plant_ext("Reference/Insurance renewals.md", "2026-02-26 09:10", """
# Insurance renewals

| Policy | What | Renews | Premium |
|---|---|---|---|
| HM-5530917 | Studio liability, Halden Mutual | March 2 | $612 |

Equipment cover is a rider on the same policy: the laptop and plotter are insured for up to $9,800 in total. Claims line is on the back of the card.
""")

plant_ext("Reference/Recipe - Peanut noodles.md", "2026-03-14 18:55", """
---
tags: [recipe, noodles]
---
# Peanut noodles

Sauce: 3 tbsp peanut butter, 2 tbsp soy, 1 tbsp black vinegar, a spoon of chilli crisp, hot water to loosen.

Boil the noodles for 3 minutes 40 seconds, rinse under cold water so they don't clump, then toss while still slightly warm.
""")

plant_ext("Reference/Recipe - Plum crumble.md", "2025-10-12 17:20", """
---
tags: [recipe, dessert]
---
# Plum crumble

Halve the plums, toss with sugar and a pinch of cardamom, and cover with a topping of oats, butter and flour.

The plums come from the Hennessy stall at the Sunday market. Bake until the juice bubbles at the edges.
""")

plant_ext("Journal/2026-08-29.md", "2026-08-29 22:30", """
Booked Mum's birthday dinner at Sorrel & Pine for Saturday the 12th, a table for five at 19:30. She picked it herself, which never happens.

Gave up on the fig tree. Cut it back hard and hoped.

#journal
""")

plant_ext("Reading/Counting Letters - Maddox.md", "2026-04-22 20:00", """
---
tags: [reading, typography]
author: Isla Maddox
rating: 3
---
# Counting Letters (Maddox)

Uneven but useful. Maddox sets small-screen text with a leading of 1.45 and says anything tighter than 1.3 makes paragraphs look like walls.

You can skip chapter 7. It is forty pages on the history of kerning and none of it changes how you work.
""")

plant_ext("Reading/Contracts for Creatives - notes.md", "2026-07-05 19:10", """
---
tags: [reading, contracts]
author: Bea Lindo
rating: 4
---
# Contracts for Creatives (Lindo)

Clause to steal: a kill fee of 30% if the client cancels after sketches have been shown.

Lindo's suggested payment schedule is 40/40/20: start, midpoint approval, delivery. I have been doing 50/50, which leaves me exposed at the back end.
""")

plant_ext("Reference/Obsidian vault tweaks.md", "2026-05-03 11:55", """
# Obsidian vault tweaks

- Quick switcher is remapped to Cmd-Shift-O because Cmd-O collided with the file picker.
- New notes land in Inbox, daily notes use Templates/daily.
- Spellcheck language: English (UK).
""")

plant_ext("Reference/Export presets.md", "2026-06-09 15:35", """
# Export presets

Web: PNG at 2x, WebP at quality 74. App icons at 3x.

Social: Instagram portrait posts are 1080 x 1350, stories 1080 x 1920.
""")

plant_ext("Reference/Mac shortcuts.md", "2025-12-02 08:20", """
# Mac shortcuts I forget

Screenshot a region to the clipboard: Ctrl-Cmd-Shift-4.
Karabiner remap: Caps Lock is Escape when tapped and Control when held.
Force quit: Cmd-Option-Esc.
""")

plant_ext("Reference/Network.md", "2026-02-15 12:40", """
# Network

Router admin page: 192.168.7.1, login is on the sticker under the unit.
Visitor wifi is called Dovecote-Guest and the password changes every quarter.
Printer has a fixed address, .48 at the end.
""")

plant_ext("Meetings/2026-04-08 Bramble launch planning.md", "2026-04-08 17:00", """
---
tags: [meeting, bramble]
created: 2026-04-08
---
# Bramble launch planning

**Attendees:** Maren, Josie Hart, Rafe from the shop

## Notes
- The launch event is at the old tram depot on May 30. Capacity is 120 so we need an RSVP list.
- Caterer is Olwen's Table at $14 a head, drinks separate.
- Josie wants the new jersey on a mannequin at the door.

## Action items
- [ ] Draft the invite
""")

plant_ext("Meetings/2026-06-17 Marlow range review.md", "2026-06-17 16:15", """
---
tags: [meeting, marlow]
created: 2026-06-17
---
# Marlow range review

**Attendees:** Maren, Beatrix Ng

## Notes
- The new breakfast blend is called Copper Kettle. Beatrix vetoed the first two names for sounding like a pub.
- Shelf test runs in 12 stores starting August 4.

## Action items
- [ ] Send Beatrix the Copper Kettle carton roughs
""")

plant_ext("Meetings/2026-03-31 Juniper site review.md", "2026-03-31 15:50", """
---
tags: [meeting, juniper]
created: 2026-03-31
---
# Juniper site review

**Attendees:** Maren, Odalys Ferreira, Lena

## Notes
- Odalys wants the appointment form to ask for pet weight, because the vets size the sedation dose from it.
- Launch is set for April 21, a Tuesday, so no content changes after the 14th.
""")

plant_ext("Meetings/2026-10-06 Studio ops sync.md", "2026-10-06 15:30", """
---
tags: [meeting, studio-ops]
created: 2026-10-06
---
# Studio ops sync

## Notes
- Replace the grinder in the kitchen corner. Budget is $180, and Maren picks it by Friday.
- Marisol, the new cleaner, comes Tuesdays at 6pm after the shop below closes.
""")

plant_ext("People/Hanne Lindqvist.md", "2025-11-19 10:25", """
# Hanne Lindqvist

Paper rep at the Lindqvist firm. Prefers phone calls to email. Her direct line ends 0147.

Birthday is March 9; she likes the little almond pastries from the bakery near the depot.
""")

plant_ext("People/Lena Vasquez.md", "2026-03-09 13:10", """
---
tags: [people, dev]
---
# Lena Vasquez

Freelance developer I hand finished sites to. She builds everything in Astro and bills $70 an hour.

She is away all of July every year, so anything that needs her has to ship by mid-June.
""")

plant_ext("People/Walter Achterberg.md", "2026-09-11 18:35", """
---
tags: [people, mentor]
---
# Walter Achterberg

Retired studio owner, my sounding board. We meet at Café Lindenhof on the first Monday of each quarter, he pays for the first round and I pay for the second.

Last time he told me to double my day rate for new clients and to stop explaining the price. "The more you explain the less they believe it."
""")

plant_ext("Ideas/Portfolio rebuild.md", "2026-10-19 09:55", """
---
tags: [idea, portfolio]
status: maybe
---
# Portfolio rebuild

Host it on Netlify, static, built with Astro like Lena's other sites. Show no more than 6 projects; clients want to see range, not volume.

Case studies to lead with: Tidewater signage, Marlow cartons, Kestrel app.
""")

plant_ext("Reference/Domain renewals.md", "2026-07-28 08:45", """
# Domain renewals

maren-studio.example renews August 14 at Quillfeather Domains, $18.99 a year. Auto-renew is on but the card on file expires in July, so update it before then.

Last receipt: ORD-77412, renewal for the year.
""")

plant_ext("Inbox/Call back the framer.md", "2026-10-22 07:50", """
Call Tomek at Alder Frameworks about the walnut frames for the Marlow shelf display. He said 15 days but promised 12 on the phone. Ask for quote QT-3306 in writing.
""")

plant_ext("Reference/Time tracker setup.md", "2026-04-18 12:05", """
---
tags: [reference, tools]
---
# Time tracker setup

Clockwork is the hours app. It rounds up to the nearest 15 minutes per entry, and the weekly summary email lands on Mondays at 6:30.
""")

plant_ext("Journal/2026-09-30.md", "2026-09-30 21:45", """
Update: changed Clockwork to round up to the nearest 6 minutes, so short calls stop turning into quarter hours. Should have done this a year ago.

Took the bike to Spoke & Sons for a new chain, $48, and they adjusted the brakes for free.

#journal
""")

assert len(PLANTS_EXT) == 43, len(PLANTS_EXT)

# --------------------------------------------------------------------------------------
# Decoys in .trash/ (must be ignored by indexers) and attachments
# --------------------------------------------------------------------------------------

TRASH = {
    "Harbor permit draft.md": "# Harbor permit draft\n\nOld guess, wrong: approval expected by April 30. Delete this.\n",
    "Kestrel rate idea.md": "# Kestrel rate idea\n\nFloated a day rate of $990 in my head. Never sent. Trash.\n",
}
DECOY_STRINGS = ["April 30", "$990"]

# 1x1 transparent PNG
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

# --------------------------------------------------------------------------------------
# Golden cases
# --------------------------------------------------------------------------------------
# (kind, question, gold paths, answer_contains, stale_notes)

P = "Projects/"
CASES: list[dict] = []


def case(kind, question, gold, contains, stale=None):
    CASES.append(dict(kind=kind, question=question, gold=gold, contains=contains, stale=stale))


case("direct", "When is the Harbor sign permit approval expected?",
     [P + "Harbor/permit-timeline.md"], ["March 23"])
case("direct", "Which hosting plan did we pick for the Juniper Veterinary website, and what does it cost per month?",
     [P + "Juniper/hosting-setup.md"], ["Fernhost Standard", "$27.40"])
case("direct", "What port does the Kestrel staging API listen on?",
     [P + "Kestrel/api-notes.md"], ["port 8421"])
case("direct", "What cap on dieline revisions did Marlow agree to at the packaging kickoff?",
     ["Meetings/2026-03-12 Marlow packaging kickoff.md"], ["cap of 4 dieline revisions"])
case("direct", "What late fee do I charge on overdue client invoices?",
     ["Reference/Client invoice SOP.md"], ["1.5% late fee"])
case("direct", "What line length does Atomic Interfaces recommend for body text?",
     ["Reading/Atomic Interfaces - notes.md"], ["62 characters"])
case("direct", "How long should the sourdough loaf proof in the fridge?",
     ["Reference/Sourdough starter schedule.md"], ["19 hours"])
case("direct", "What working name did I pick for the studio newsletter?",
     ["Ideas/Studio newsletter.md"], ["Overset Weekly"])

case("paraphrase", "When is the pipe repair guy coming?",
     [P + "House/plumbing-schedule.md"], ["14th of November"])
case("paraphrase", "How long will I be without my computer while it gets fixed?",
     ["Journal/2026-09-18.md"], ["6-day turnaround"])
case("paraphrase", "What's the fastest way to reach the owner of the woodworking client?",
     ["Clients/Halvorsen & Daughters (furniture).md"], ["Signal within minutes"])
case("paraphrase", "What fraction of client income do I stash for the government each time I get paid?",
     ["Reference/Quarterly tax set-aside.md"], ["28%"])
case("paraphrase", "Why did we cut the night theme from the first release of the logistics app?",
     ["Meetings/2026-05-07 Kestrel retro.md"], ["contrast audit failed"])
case("paraphrase", "What does the book suggest about showing empty space on a quote?",
     ["Reading/The Quiet Grid - Brandvold.md"], ["whitespace as a line item"])
case("paraphrase", "Who is the potter I wanted to invite on the show?",
     ["Ideas/Podcast guest wishlist.md"], ["Priya Raman"])
case("paraphrase", "How often do I refresh the copy of my files kept at my mother's place?",
     ["Reference/Backup routine.md"], ["first Sunday"])

case("entity", "What is invoice INV-4471 for?",
     [P + "Harbor/invoice-log.md"], ["window decals"])
case("entity", "What is ticket KST-318 about?",
     [P + "Kestrel/ticket-index.md"], ["offline sync crash"])
case("entity", "Which supplier has the code SUP-0912?",
     ["Reference/Suppliers.md"], ["Lindqvist Print"])
case("entity", "What does build 3.8.2-rc4 add?",
     [P + "Kestrel/release-notes.md"], ["biometric login"])

case("multi_note", "What do Tidewater's invoices INV-4471 and INV-4502 add up to?",
     [P + "Harbor/invoice-log.md", "Clients/Tidewater Bakery.md"], ["$3,260", "$1,980"])
case("multi_note", "How long is the lead time for the printer that makes the Marlow cartons?",
     [P + "Marlow/print-spec.md", "Reference/Suppliers.md"], ["SUP-0912", "11 business days"])
case("multi_note", "What is the total plumbing bill, deposit plus final balance?",
     [P + "House/plumbing-schedule.md", P + "House/payments-log.md"], ["$380", "$1,240.75"])

case("updated", "What is my current day rate for Kestrel?",
     [P + "Kestrel/rate-change.md"], ["$720"], ["Clients/Kestrel Freight.md"])
case("updated", "Where is the quarterly review held now?",
     ["Meetings/2026-08-20 Studio ops sync.md"], ["Alder Street library"],
     ["Meetings/2026-01-15 Studio ops sync.md"])

# no_answer: plausible questions the vault does not answer. The absent terms are NOT
# written to golden.json (format is fixed); the self-check reads them from here.
NO_ANSWER: list[tuple[str, list[str]]] = [
    ("What budget did we approve for the Lisbon trade show booth?", ["Lisbon", "trade show"]),
    ("What is the invoice number for the Okonkwo Dental rebrand?", ["Okonkwo", "Dental"]),
    ("Which supplier did we choose for the wedding stationery order?", ["wedding", "stationery order"]),
    ("What port does the Grafana dashboard run on?", ["Grafana"]),
    ("When is the permit hearing for the Calloway rooftop deck?", ["Calloway", "rooftop"]),
]
for _q, _terms in NO_ANSWER:
    case("no_answer", _q, [], [])


# --------------------------------------------------------------------------------------
# Extension golden cases (v2): appended after the original 30, only emitted at N >= 100.
# 81 new answerable cases: 43 paraphrase, 14 direct, 12 entity, 8 multi_note, 4 updated.
# --------------------------------------------------------------------------------------

CASES_EXT: list[dict] = []


def case2(kind, question, gold, contains, stale=None):
    CASES_EXT.append(dict(kind=kind, question=question, gold=gold, contains=contains, stale=stale))


# ---- paraphrase (38): no distinctive word shared with the gold note (checked by --self-check) ----
case2("paraphrase", "How does the spectacles retailer's creative boss like to review proofs?",
      ["Clients/Fennick Optics.md"], ["A3 printouts"])
case2("paraphrase", "When can shoppers return to the third branch after the building work?",
      ["Meetings/2026-09-22 Fennick catch-up.md"], ["November 3"])
case2("paraphrase", "What day of the week does the spirits maker run its filling session?",
      ["Clients/Salt Row Distillery.md"], ["first Thursday of the month"])
case2("paraphrase", "Which person at the spirits maker has trouble telling certain hues from one another?",
      ["Clients/Salt Row Distillery.md"], ["red-green colour blind"])
case2("paraphrase", "Why does the cream-and-yoghurt company's accounts site keep returning my paperwork?",
      ["Clients/Greywell Dairy.md"], ["purchase order number"])
case2("paraphrase", "How frequently does the firm's decision-maker want progress reports?",
      ["Clients/Ostrander Architects.md"], ["Fridays at 4"])
case2("paraphrase", "How many books will the first batch contain for the fiction publisher?",
      ["Projects/Quillon/cover-series-notes.md"], ["2,400 copies"])
case2("paraphrase", "What kind of fuse is needed for the new stove top?",
      ["Projects/House/electrician-visit.md"], ["40-amp breaker"])
case2("paraphrase", "Where do I keep the replacement cells for the fire detector?",
      ["Journal/2026-10-03.md"], ["drawer under the kettle"])
case2("paraphrase", "What is the duration of each isometric stay in my rehab routine?",
      ["Health/Physio plan.md"], ["30 seconds"])
case2("paraphrase", "When do I next see my family doctor?",
      ["Health/Annual checkup.md"], ["January 27"])
case2("paraphrase", "What is my daily dose of the sunshine pill over the dark months?",
      ["Health/Annual checkup.md"], ["2000 IU"])
case2("paraphrase", "How long is the sightseeing pass valid that I plan to purchase?",
      ["Travel/Tallinn trip.md"], ["48-hour"])
case2("paraphrase", "Where can I find the emergency way into my mother's home?",
      ["Travel/Mum visit.md"], ["blue heron pot"])
case2("paraphrase", "What annual yield does the reserve I keep for the government earn?",
      ["Reference/Business accounts.md"], ["3.9%"])
case2("paraphrase", "On what date does the plastic stop counting new purchases for the cycle?",
      ["Reference/Business accounts.md"], ["closes on the 17th"])
case2("paraphrase", "Which sour liquid goes into the satay-style dressing?",
      ["Reference/Recipe - Peanut noodles.md"], ["black vinegar"])
case2("paraphrase", "Which warm seasoning goes onto the stewed fruit before baking?",
      ["Reference/Recipe - Plum crumble.md"], ["pinch of cardamom"])
case2("paraphrase", "Which restaurant did my mother choose for the celebration meal?",
      ["Journal/2026-08-29.md"], ["Sorrel & Pine"])
case2("paraphrase", "What spacing between baselines does the writer favour on handsets?",
      ["Reading/Counting Letters - Maddox.md"], ["leading of 1.45"])
case2("paraphrase", "Which section of the volume can I leave unread?",
      ["Reading/Counting Letters - Maddox.md"], ["skip chapter 7"])
case2("paraphrase", "What share of the price should I demand if a buyer walks away once roughs have been presented?",
      ["Reading/Contracts for Creatives - notes.md"], ["kill fee of 30%"])
case2("paraphrase", "Which shortcut combination lets me hop to any page by typing its name?",
      ["Reference/Obsidian vault tweaks.md"], ["Cmd-Shift-O"])
case2("paraphrase", "What happens when I briefly press the key that normally capitalises everything?",
      ["Reference/Mac shortcuts.md"], ["Caps Lock is Escape"])
case2("paraphrase", "Which wireless name do people dropping by connect to?",
      ["Reference/Network.md"], ["Dovecote-Guest"])
case2("paraphrase", "Where will the bicycle company's opening celebration take place?",
      ["Meetings/2026-04-08 Bramble launch planning.md"], ["old tram depot"])
case2("paraphrase", "What name did they settle on for the morning tea?",
      ["Meetings/2026-06-17 Marlow range review.md"], ["Copper Kettle"])
case2("paraphrase", "What additional detail does the animal practice want on the reservation sheet?",
      ["Meetings/2026-03-31 Juniper site review.md"], ["pet weight"])
case2("paraphrase", "When does the person who tidies the premises arrive each week?",
      ["Meetings/2026-10-06 Studio ops sync.md"], ["Tuesdays at 6pm"])
case2("paraphrase", "On what date was my sales contact for boards and labels born?",
      ["People/Hanne Lindqvist.md"], ["March 9"])
case2("paraphrase", "Which website framework does my contract coder use?",
      ["People/Lena Vasquez.md"], ["in Astro"])
case2("paraphrase", "During which month is my contract coder unavailable?",
      ["People/Lena Vasquez.md"], ["away all of July"])
case2("paraphrase", "Where do I get together with my adviser every few months?",
      ["People/Walter Achterberg.md"], ["Café Lindenhof"])
case2("paraphrase", "How many pieces should my online showroom feature at most?",
      ["Ideas/Portfolio rebuild.md"], ["no more than 6"])
case2("paraphrase", "When does my web address registration lapse?",
      ["Reference/Domain renewals.md"], ["August 14"])
case2("paraphrase", "What is the hourly price of my bean-counter's assistance with the ledgers?",
      ["Reference/Accountant.md"], ["$85 an hour"])
case2("paraphrase", "How much does protection against lawsuits cost me annually?",
      ["Reference/Insurance renewals.md"], ["$612"])
case2("paraphrase", "When does the digest of my logged effort get delivered?",
      ["Reference/Time tracker setup.md"], ["Mondays at 6:30"])
case2("paraphrase", "What is the smallest charge the sparky levies for coming out?",
      ["Projects/House/electrician-visit.md"], ["$95"])
case2("paraphrase", "How much extra turnaround does metallic stamping on the novel's backbone add?",
      ["Projects/Quillon/cover-series-notes.md"], ["6 working days"])
case2("paraphrase", "How long should the strands bubble away in the pot?",
      ["Reference/Recipe - Peanut noodles.md"], ["3 minutes 40 seconds"])
case2("paraphrase", "When do I sit down with my bean-counter for the annual review?",
      ["Reference/Accountant.md"], ["January 13 at 2pm"])
case2("paraphrase", "From which bay does the rail service to my mother's depart?",
      ["Travel/Mum visit.md"], ["Platform 4"])

# ---- direct (14) ----
case2("direct", "What finish did Dorothea ask for on the Harbor Street shop decals?",
      ["Meetings/2026-09-22 Fennick catch-up.md"], ["matte, not gloss"])
case2("direct", "What minimum height does the government warning text need on the Salt Row labels?",
      ["Clients/Salt Row Distillery.md"], ["2.5 mm"])
case2("direct", "Which Pantone is the Greywell green?",
      ["Clients/Greywell Dairy.md"], ["7731 C"])
case2("direct", "What colour did I paint the hallway?",
      ["Journal/2026-10-03.md"], ["Oyster Shell"])
case2("direct", "What size running shoes did I buy after the gait test?",
      ["Journal/2026-10-14.md"], ["size 44"])
case2("direct", "What is the maximum insured value of the laptop and plotter?",
      ["Reference/Insurance renewals.md"], ["$9,800"])
case2("direct", "What payment split does Lindo suggest in Contracts for Creatives?",
      ["Reading/Contracts for Creatives - notes.md"], ["40/40/20"])
case2("direct", "What address do I type to open the router admin page?",
      ["Reference/Network.md"], ["192.168.7.1"])
case2("direct", "How much per person does the caterer charge for the Bramble launch?",
      ["Meetings/2026-04-08 Bramble launch planning.md"], ["$14 a head"])
case2("direct", "How many stores will run the Marlow shelf test?",
      ["Meetings/2026-06-17 Marlow range review.md"], ["12 stores"])
case2("direct", "Where did I get the bike chain replaced?",
      ["Journal/2026-09-30.md"], ["Spoke & Sons"])
case2("direct", "What did Walter say I should do with my day rate for new clients?",
      ["People/Walter Achterberg.md"], ["double my day rate"])
case2("direct", "How does Hanne's direct line number end?",
      ["People/Hanne Lindqvist.md"], ["ends 0147"])
case2("direct", "What size are Instagram portrait posts in my export presets?",
      ["Reference/Export presets.md"], ["1080 x 1350"])

# ---- entity (12) ----
case2("entity", "What is job OA-26-044?",
      ["Clients/Ostrander Architects.md"], ["Chapel restoration signage"])
case2("entity", "Who is supplier SUP-1177?",
      ["Projects/Bramble/frame-decals.md"], ["Strand & Reeve"])
case2("entity", "Which paper stock is PS-233?",
      ["Reference/Paper stocks.md"], ["Ardent White"])
case2("entity", "What is paper stock PS-248?",
      ["Reference/Paper stocks.md"], ["Birchline Kraft"])
case2("entity", "What is policy HM-5530917 for?",
      ["Reference/Insurance renewals.md"], ["Studio liability"])
case2("entity", "What is quote QT-3306 for?",
      ["Inbox/Call back the framer.md"], ["walnut frames"])
case2("entity", "What is invoice INV-4933 for?",
      ["Clients/Salt Row Distillery.md"], ["Rye label"])
case2("entity", "What is ticket JNP-412 about?",
      ["Projects/Juniper/ticket-index.md"], ["duplicate emails"])
case2("entity", "What is ticket JNP-427 about?",
      ["Projects/Juniper/ticket-index.md"], ["Kellner Street"])
case2("entity", "What does Juniper release v1.4.0 add?",
      ["Projects/Juniper/changelog.md"], ["SMS appointment reminders"])
case2("entity", "Why was proof PRF-118 rejected?",
      ["Projects/Quillon/proof-log.md"], ["spine width"])
case2("entity", "What is order ORD-77412?",
      ["Reference/Domain renewals.md"], ["renewal for the year"])

# ---- multi_note (8): gold has 2 notes, each contributes one answer ----
case2("multi_note", "What do the Salt Row gin label invoice and the Greywell milk carton invoice come to together?",
      ["Clients/Salt Row Distillery.md", "Clients/Greywell Dairy.md"], ["$2,750", "$1,430"])
case2("multi_note", "What do one night at the Tallinn guesthouse and the 48-hour city card cost in total?",
      ["Ideas/Autumn trip.md", "Travel/Tallinn trip.md"], ["€92 a night", "€55"])
case2("multi_note", "How much more per hour does the accountant charge for bookkeeping than the freelance developer bills?",
      ["Reference/Accountant.md", "People/Lena Vasquez.md"], ["$85 an hour", "$70 an hour"])
case2("multi_note", "When is the Juniper site going live, and which release brought in the text-message reminders?",
      ["Meetings/2026-03-31 Juniper site review.md", "Projects/Juniper/changelog.md"], ["April 21", "v1.4.0"])
case2("multi_note", "What day and time does the Ostrander weekly email go out, and which day does Salt Row bottle each month?",
      ["Clients/Ostrander Architects.md", "Clients/Salt Row Distillery.md"], ["Fridays at 4", "first Thursday of the month"])
case2("multi_note", "When does the company card cycle close, and when is the year-end meeting with Nuala?",
      ["Reference/Business accounts.md", "Reference/Accountant.md"], ["closes on the 17th", "January 13 at 2pm"])
case2("multi_note", "Which platform does my train to Mum's leave from, and where is her birthday dinner?",
      ["Travel/Mum visit.md", "Journal/2026-08-29.md"], ["Platform 4", "Sorrel & Pine"])
case2("multi_note", "Is Lena away when the Juniper site launches?",
      ["People/Lena Vasquez.md", "Meetings/2026-03-31 Juniper site review.md"], ["away all of July", "April 21"])

# ---- updated (4): the gold note says "Update" and is newer than the stale note ----
case2("updated", "On which day of the month does Fennick pay supplier bills now?",
      ["Meetings/2026-09-22 Fennick catch-up.md"], ["5th of the month"], ["Clients/Fennick Optics.md"])
case2("updated", "How coarse is the rounding on my hours app these days?",
      ["Journal/2026-09-30.md"], ["nearest 6 minutes"], ["Reference/Time tracker setup.md"])
case2("updated", "When is my weekly physio session now?",
      ["Journal/2026-10-14.md"], ["Tuesdays at 7:40"], ["Health/Physio plan.md"])
case2("updated", "What time does my plane to Estonia leave now?",
      ["Travel/Tallinn trip.md"], ["11:50"], ["Ideas/Autumn trip.md"])

# Extra strings the filler must never contain at N >= 100, on top of every answer_contains:
# the plants' identifiers (found by pattern below) and a few names and versions.
RESERVED_EXT: list[str] = sorted(set(
    [s for c in CASES_EXT for s in c["contains"]] +
    ["1.4.0", "1.3.2", "v1.3.2", "Lena Vasquez", "Tamsin Okoye", "Wilf Pardoe", "Nuala Brennan",
     "Dorothea Lund", "Piet Lund", "Anneke Voorhees", "Isla Maddox", "Bea Lindo", "$2,750", "$1,900",
     "$1,430", "Kellner", "Larchmont", "Halden", "Quillfeather", "Dunmore", "Stride Lab", "Hennessy"] +
    [m for _p, _w, t in PLANTS_EXT for m in re.findall(r"\b[A-Z]{2,4}-\d[\w-]*", t)]))

# --------------------------------------------------------------------------------------
# Near-miss prompts (nearmiss.json): they reuse the vault's vocabulary but the vault cannot
# answer them. Three families: advice or how-to, drafting requests, and questions about things
# that do not exist in the vault (other clients, ports, codes). Every number or identifier in
# a prompt must be absent from the vault (--self-check asserts it).
# --------------------------------------------------------------------------------------

NEARMISS: list[str] = [
    # generic how-to and advice in the vault's domains
    "How do I decide whether a small clinic website needs a staging site or only daily backups?",
    "What is a sensible way to version a staging API for a mobile app?",
    "How do I keep a sourdough starter alive while I am away for two weeks?",
    "How can I get a shop-front sign permit approved faster?",
    "What is the best way to push a supplier's lead time down without annoying them?",
    "How often should I actually test restoring from a backup?",
    "What are the trade-offs between net 14 and net 30 invoice terms for a small studio?",
    "What should the first issue of a studio newsletter cover?",
    # drafting and writing requests that use vault words
    "Draft a short email asking a plumber to confirm a quote for replacing a kitchen shutoff valve.",
    "Write a thank-you note to a supplier whose print proof arrived two days early.",
    "Draft a LinkedIn post announcing a monthly newsletter from a small design studio.",
    "Write a one-paragraph project brief for a tea carton packaging redesign.",
    "Draft a polite overdue reminder for invoice INV-1234.",
    "Write alt text for a photo of a blade sign above a bakery window.",
    "Draft meeting notes for a kickoff call about moving a clinic website to a new hosting plan.",
    # things that do not exist in the vault
    "What port does the Kestrel admin dashboard listen on, is it 2087?",
    "What is invoice INV-0420 for?",
    "Which supplier has the code SUP-4410?",
    "What is ticket KST-1204 about?",
    "When is the review date for permit BP-0417?",
    "Which hosting plan did we pick for the Bramble Cycles shop site?",
    "What hydration percentage do I use for the sourdough?",
    "Which email tool did I pick for the newsletter?",
    "Which model is the blue backup drive?",
    "What does the Marlow Tea logo contract say about usage rights?",
    # coding and tooling prompts that use project names
    "Write a Python script that pages through /v2/orders on the Kestrel API and saves each page to disk.",
    "How do I set up a GitHub Action that deploys the Juniper site to Netlify on every push?",
    "Write a bash script that keeps the last seven Time Machine style snapshots of the Harbor folder.",
    "Write a SQL query that sums open invoices by client for the Tidewater account.",
    "Write a CSS rule that keeps a blade sign preview card at a 3:1 aspect ratio.",
    "How do I add a bearer token header to a fetch call against the Kestrel staging API?",
    "Write a regex that matches invoice numbers shaped like INV-0420.",
    "Explain how to restore a Time Machine backup onto a new laptop.",
]

# Per-kind caps for small vaults (N < 100): fewer cases, same plants.
SMALL_N_CAPS = {"direct": 3, "paraphrase": 3, "entity": 2, "multi_note": 1, "updated": 1, "no_answer": 2}
SMALL_N_LIMIT = 100


def build_golden(n_notes: int) -> dict:
    caps = dict(SMALL_N_CAPS) if n_notes < SMALL_N_LIMIT else None
    seen: dict[str, int] = {}
    cases = []
    for c in (CASES if n_notes < SMALL_N_LIMIT else CASES + CASES_EXT):
        k = c["kind"]
        seen[k] = seen.get(k, 0) + 1
        if caps is not None and seen[k] > caps[k]:
            continue
        d = {
            "id": f"g{len(cases) + 1:03d}",
            "question": c["question"],
            "gold": list(c["gold"]),
            "expect_none": k == "no_answer",
            "kind": k,
            "answer_contains": list(c["contains"]),
        }
        if k == "updated":
            d["stale_notes"] = list(c["stale"])
        cases.append(d)
    return {"version": 1, "cases": cases}


# Strings the filler must never contain, so a lookup is unambiguous. Everything the
# golden set asserts, plus plant-specific identifiers and the trash decoys.
RESERVED_BASE: list[str] = sorted(set(
    [s for c in CASES for s in c["contains"]] + DECOY_STRINGS +
    ["INV-4471", "INV-4502", "INV-4452", "INV-4488", "SP-2291", "KST-318", "KST-301",
     "KST-322", "KST-340", "SUP-0912", "SUP-0788", "SUP-1034", "3.8.2-rc4", "3.8.1", "3.7.9",
     "8421", "Redwater", "Fernhost", "Cloudpine", "Brindle Cloud", "Dale Whitcombe",
     "Lindqvist", "Nordhavn", "Overset", "Kiln & Co",
     "Alder", "Cannery", "Odalys", "Ruth Abara", "Ingrid", "Karin", "Beatrix", "Hanne",
     "$215", "$640", "$149", "$64.10", "$210.00"]))
RESERVED: list[str] = list(RESERVED_BASE)  # generate() widens it with RESERVED_EXT from N >= 100

# --------------------------------------------------------------------------------------
# Word pools. Everything is invented. Terms used by no_answer questions
# (Lisbon, Okonkwo, wedding, Grafana, Calloway, rooftop, trade show) must not appear.
# --------------------------------------------------------------------------------------

# project -> (client, what it is)
PROJECTS = {
    "Harbor": ("Tidewater Bakery", "storefront signage and window vinyl"),
    "Juniper": ("Juniper Veterinary", "clinic website"),
    "Kestrel": ("Kestrel Freight", "warehouse handheld app"),
    "Marlow": ("Marlow Tea", "tea carton packaging"),
    "Quillon": ("Quillon Press", "novel cover series"),
    "Bramble": ("Bramble Cycles", "brand refresh"),
    "House": ("the house", "kitchen renovation"),
}
WORK_PROJECTS = ["Harbor", "Juniper", "Kestrel", "Marlow", "Quillon", "Bramble"]
CLIENTS_BASE = ["Tidewater Bakery", "Juniper Veterinary", "Kestrel Freight", "Marlow Tea", "Quillon Press",
                "Bramble Cycles", "Northgate Credit Union"]

FIRST = ["Amara", "Callum", "Esme", "Farid", "Greta", "Jonas", "Kaveh", "Lena", "Mateo", "Nadia", "Oskar",
         "Petra", "Quentin", "Rosa", "Søren", "Tomasz", "Ulla", "Vikram", "Wren", "Zoë", "Mélanie", "Łukasz",
         "Imogen", "Dmitri", "Chiara", "Bashir", "Noor", "Elif", "Tobias", "Marguerite", "Anselm", "Yusra"]
LAST = ["Osei", "Brandt", "Castellano", "Delacroix", "Eriksen", "Fofana", "Grimaldi", "Haddad", "Ivanova",
        "Jorgensen", "Kowalczyk", "Lindgren", "Moreau", "Nakamura", "Ortega", "Pryce", "Quist", "Rosenthal",
        "Sandoval", "Tran", "Underhill", "Vasquez", "Whitlock", "Yamada", "Zielinski", "Abernathy", "Bello"]

DELIVERABLES = ["logo lockup", "homepage mockup", "packaging dieline", "style guide", "icon set", "brand deck",
                "onboarding flow", "signage drawing", "cover comps", "email template", "pitch deck",
                "photo shot list", "menu layout", "wayfinding plan", "type specimen", "colour palette",
                "launch checklist", "social template pack", "label artwork", "press kit"]
TASKS = ["cleaned up the component library", "reworked the grid on the main layout", "exported the final assets",
         "redlined the print proof", "rewrote the project brief", "tightened the colour palette",
         "built the review deck", "tested the checkout flow on a phone", "sorted out font licensing",
         "drafted the scope document", "put together a clickable prototype", "reorganised the shared drive",
         "retouched the product photos", "wrote handoff notes for the developer", "kerned the display headline",
         "compared three paper stocks", "fixed the baseline grid", "audited the contrast ratios"]
THINGS = ["buying the new monitor", "renewing the font licence", "hiring a part-time assistant",
          "moving to a bigger desk", "upgrading the plotter", "switching invoicing tools",
          "taking on a second retainer", "redoing the portfolio site", "a proper holiday in spring",
          "raising rates for new clients", "a standing desk", "the second printer"]
TOOLS = ["Figma", "Illustrator", "Affinity Publisher", "Lightroom", "Notion", "InDesign", "Procreate",
         "Sketch", "Blender", "Keynote", "Photoshop", "Glyphs", "Obsidian"]
SUPPLIERS = ["Nordic Ink Works", "Marlen Paper Co", "Fenwick Signs", "Harrow & Pike Print", "Copperline Textiles",
             "Ashgrove Press", "Tallis Packaging", "Brightwater Labels", "Oakham Frames", "Sundial Screen Print",
             "Verity Board Mill", "Quarry Lane Letterpress", "Hollis Vinyl", "Pennant Display", "Larkspur Paper",
             "Westmere Cartons", "Ironside Fabrication", "Kiteley Digital Print"]
SUPPLIER_SUFFIX = ["Print", "Paper", "Supply", "Works", "Press", "Signs", "Board", "Labels"]
RECIPES = ["lentil soup", "shakshuka", "banana bread", "miso salmon", "focaccia", "chana masala",
           "overnight oats", "pesto pasta", "lemon bars", "congee", "tomato tart", "black bean chili"]
AREAS = ["studio", "finance", "admin", "client", "design", "dev", "ops", "writing", "home", "health"]

BOOK_ADJ = ["Honest", "Quiet", "Elastic", "Patient", "Hidden", "Sturdy", "Modest", "Careful", "Ordinary",
            "Visible", "Slow", "Plain", "Generous", "Restless"]
BOOK_NOUN = ["Interface", "Margin", "Craft", "Brief", "Studio", "Proof", "Alphabet", "Ledger", "Compass",
             "Sketchbook", "Pattern", "Workshop", "Contract", "Process"]
AUTHORS = ["Halvard", "Mbeki", "Okafor-Lind", "Petrova", "Sallow", "Teague", "Voss", "Wadsworth",
           "Yilmaz", "Zeller", "Arden", "Bishara", "Corrigan", "Dahl"]
ARTICLE_HEADS = ["Why clients ask for the logo bigger", "A case for boring navigation", "Pricing by value, honestly",
                 "What a design critique is for", "The hidden cost of custom illustration", "Saying no to spec work",
                 "Variable fonts without the hype", "Writing a proposal people read", "Print is not dead, it is slow",
                 "How small studios survive winter", "Colour blindness and brand palettes", "The two-hour client call"]

INSIGHTS = [
    "Constraints are the brief; the earlier you name them, the better the work.",
    "Most feedback is about a feeling, so ask what the reader should feel before changing any pixels.",
    "A style guide that nobody opens is a PDF with good intentions.",
    "Slow decisions cost more than wrong ones, because everything downstream waits.",
    "Show fewer options, and say which one you would pick.",
    "Rates are a signal; a low one tells the client the work is easy.",
    "Good hierarchy lets someone skim a page and still get the point.",
    "Write down the reason for every decision, future you will not remember it.",
    "Revisions without a cap quietly become the project.",
    "A deadline with no consequence is a suggestion.",
    "Restraint is easier to defend than decoration.",
    "Paper choice changes what the words mean, so choose it before the layout is final.",
    "Clients rarely remember the process, but they remember being surprised.",
    "Consistency beats cleverness in anything that has to be maintained.",
    "People trust a number that has a reason next to it.",
]

JOURNAL_PERSONAL = [
    "Ran 5k along the river, legs felt heavy and I walked the last bit.",
    "Cooked {recipe} for dinner and made far too much.",
    "Called Mum, she is thinking about selling the old piano.",
    "Read a few chapters of {book} before bed.",
    "Slept badly and had four coffees to compensate, which did not help.",
    "Dinner with {person2}, we talked about quitting social media again and did not mean it.",
    "Rain all day, stayed in and sorted the receipts folder.",
    "Long walk after lunch with no phone. Should do that more often.",
    "Tried the new climbing gym with a friend; forearms are done.",
    "Watered the plants, finally, and repotted the one that was gasping.",
    "Went to the market for vegetables and ended up with a lamp I did not need.",
    "Picked up the dry cleaning, posted the parcel, paid the gas bill.",
    "Early night, phone in the other room.",
    "Cycled across town for a haircut and got caught in a downpour on the way back.",
]

# Shared work sentences; slots are filled from the Slots object.
WORK_SENT = [
    "Spent most of the session on the {deliverable} for {client}; {task}.",
    "{person} signed off on the {deliverable}, with two small notes about spacing.",
    "The {proj} budget stands at {price} approved, with {price2} spent so far.",
    "Quote from {supplier} came back at {price}, a little higher than the first estimate.",
    "We agreed to revisit the {deliverable} on {date}, once {person} has seen the numbers.",
    "Invoice {inv} for {price} went out and is due in {n} days.",
    "Staging is running build {version} and answers on {port}.",
    "Still waiting on permit {permit}; nothing to do but chase.",
    "Scope crept again, so I flagged the extra {deliverable} as billable at {price}.",
    "Lost half a day because the {tool} file would not open after the update to {version}.",
    "Cut the extra round of revisions; confirmed with {person} on {date}.",
    "Budget is tight on this one, so {thing} moves to next quarter.",
    "{person} asked for a cheaper option, so I priced a smaller version at {price}.",
    "Checked the {deliverable} against the brief and wrote down {n} things to fix before it goes to {client}.",
    "Did a pass on the {tool} library, mostly renaming layers and deleting unused styles.",
    "Handoff to the developer is set for {date}; the notes are in the shared folder.",
    "Sample from {supplier} arrived. The weight is right but the colour is a shade off.",
    "{ticket} is still open and blocking the {deliverable}.",
    "Reminder to self: {person} reads email on Mondays only, so send anything important then.",
    "Moved the {deliverable} review to {date} so nobody has to rush it.",
    "Cash flow is fine this month. {inv} cleared and {inv2} is on its way.",
    "Proofread the {deliverable} twice and still found a typo in the footer.",
    "Spent an hour on the phone with {supplier}; lead time is {bd} business days, not the {n2} they promised.",
]

LINK_SENT = ["See {link} for the background.", "Related: {link}.", "This follows on from {link}.",
             "Context is in {link}.", "Compare with {link} before deciding.", "Details live in {link}."]

MEETING_KINDS = ["sprint review", "budget review", "kickoff", "design critique", "supplier call", "handoff",
                 "status call", "retro", "scope check", "review prep", "weekly check-in", "feedback round"]
MEETING_BUDGET = [
    "Budget: {price} approved for {deliverable}, {price2} already spent.",
    "Budget check: we are at {price} of {price2}. {person} wants a breakdown before the next call.",
    "Budget is the sticking point; {person} cannot go past {price} without a sign-off.",
    "Budget: no change. Still {price} for the {deliverable}, plus {price2} contingency.",
    "Budget conversation: the {deliverable} comes to {price} if we drop the extra round.",
]
MEETING_LINES = [
    "Decision: go with {supplier} for the {deliverable}, quote {price}, decided {date}.",
    "{person} will send the revised {deliverable} by {date}.",
    "Open question: can we afford {thing} inside the {price} budget?",
    "Staging for {proj} is at {version}, API answering on {port}.",
    "Invoice {inv} goes out after sign-off.",
    "Permit {permit} is with the city; no news yet.",
    "Agreed to cap feedback at {n} rounds for the {deliverable}.",
    "{person} raised concerns about the timeline; we pushed the review to {date}.",
    "{ticket} moved to next sprint.",
    "Parked: {thing}.",
]
AGENDA = ["Status of the {deliverable}", "Budget and invoices", "Open risks", "Timeline for the next two weeks",
          "Feedback from {client}", "Supplier lead times", "Anything blocking", "Handoff plan"]
ACTIONS = ["{person} to send the revised {deliverable} by {date}", "Maren to chase {supplier} for the proof",
           "Maren to update the budget sheet", "Confirm the {deliverable} sizes with {person}",
           "Send invoice {inv} after sign-off", "Book the review for {date}"]

STEPS = [
    "Open the {tool} project and check the version is {version}.",
    "Duplicate the file first and add the date to the name.",
    "Run the export preset called 'handoff' and wait for it to finish.",
    "Check the output folder against the checklist; open two files at random.",
    "Zip the folder and upload it to the client's shared drive.",
    "Send the link with a two-line summary, never the attachment.",
    "If anything fails, note the error in the project's decision log.",
    "Rename the layers using the studio convention (lowercase, hyphens).",
    "Flatten effects that the printer cannot handle.",
    "Embed fonts or outline them, depending on the supplier's rule.",
    "Confirm colour mode with the printer before exporting.",
    "Archive the working files to the dated folder and clear the desktop.",
    "Record the time spent so the next estimate is honest.",
    "Wait a day and look at it again before sending.",
]

SNIPPETS = [
    ("bash", "rsync -avh --progress ~/Studio/Active/ /Volumes/BlueDrive/Studio-backup/ --exclude '.DS_Store'"),
    ("bash", "ffmpeg -i draft.mov -vf scale=1280:-2 -crf 23 -preset slow draft-720.mp4"),
    ("bash", "npm install --save-dev vite@%VERSION%\nnpm run dev -- --port %PORTNUM%"),
    ("json", '{\n  "name": "client-site",\n  "version": "%VERSION%",\n  "scripts": { "dev": "vite --port %PORTNUM%" }\n}'),
    ("css", ":root {\n  --measure: 64ch;\n  --space: 1.5rem;\n}\narticle { max-width: var(--measure); }"),
    ("python", "from pathlib import Path\nfor f in sorted(Path('exports').glob('*.png')):\n    f.rename(f.with_name(f.stem + '-final' + f.suffix))"),
    ("yaml", "services:\n  api:\n    image: studio/api:%VERSION%\n    ports:\n      - \"%PORTNUM%:8080\""),
    ("bash", "convert hero.png -resize 1600x -quality 82 hero-web.jpg"),
    ("sql", "select client, sum(amount) from invoices where status = 'open' group by client;"),
    ("bash", "curl -s http://localhost:%PORTNUM%/health | jq .status"),
]

HOWTO_VERB = ["Export", "Archive", "Back up", "Batch rename", "Colour-proof", "Package", "Compress", "Version"]
HOWTO_OBJ = ["client logos", "web assets", "print PDFs", "photo shoots", "font files", "shared libraries",
             "invoices", "contracts", "brand guidelines", "video drafts"]
SOP_NAMES = ["Client onboarding", "Final file delivery", "Invoice follow-up", "Kickoff call", "Proofing round",
             "Retainer renewal", "Font licensing check", "Project close-out", "Quote preparation"]
PROJECT_TOPICS = ["scope", "timeline", "feedback-round", "research", "deliverables", "budget-tracker",
                  "moodboard-notes", "stakeholder-map", "qa-checklist", "handoff-notes", "decision-log",
                  "risks", "launch-plan", "retro", "content-inventory", "review-prep"]
HOUSE_TOPICS = ["contractor-quotes", "tile-options", "paint-colours", "permit-questions", "appliance-shortlist",
                "lighting-plan", "cabinet-hardware", "demolition-notes"]
IDEA_A = ["Monthly", "Quarterly", "Pay-what-you-want", "Printed", "Short", "Public", "Members-only", "Seasonal"]
IDEA_B = ["type specimen", "studio notes", "portfolio review", "brand audit", "template pack", "poster series",
          "client workshop", "sketchbook zine", "colour study", "mockup kit"]
IDEA_LINES = [
    "Could be a weekend project. Needs a landing page and a way to take payment.",
    "Talk to two past clients before spending any time on it.",
    "Price it low at first and see who bites.",
    "The risk is that it becomes another thing to maintain.",
    "Sketch the first three pieces and see if it still feels good.",
    "Probably works better as a small print run than a digital thing.",
    "Ask {person} what they would pay for this.",
]
INBOX_TEMPLATES = [
    "Call back {person}", "Check {thing}", "Receipt - {supplier}", "Link to read later",
    "Reply to {person}", "Renew {tool} licence", "Order samples", "Pick up proofs",
]
LONG_HEADS = ["Background", "Goals", "What we shipped", "Timeline", "Budget", "Stakeholders", "Early mistakes",
              "Decisions that held up", "Decisions we reversed", "Supplier notes", "Feedback patterns",
              "Technical constraints", "What the client said", "Review rounds", "Handoff", "Risks",
              "Money", "Next steps", "Open questions", "Lessons", "Tooling", "Scheduling", "Quality checks"]
TAGS_BY_KIND = {
    "journal": ["journal", "daily", "reflection", "studio"], "meeting": ["meeting", "client", "review"],
    "project": ["project", "design", "client"], "reference": ["reference", "howto", "sop", "tools"],
    "reading": ["reading", "books", "learning"], "idea": ["idea", "someday"], "inbox": ["inbox", "todo"],
    "client": ["client", "billing"], "root": ["index"],
}

# --------------------------------------------------------------------------------------
# Fact generation: per-note random specifics that never collide with the plants
# --------------------------------------------------------------------------------------


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


class Facts:
    """Invented invoice numbers, prices, ports, versions, permits, dates, suppliers.

    Every value is checked against RESERVED so a plant's fact can never occur in filler.
    """

    def __init__(self, r: random.Random):
        self.r = r

    def _ok(self, s: str) -> bool:
        for x in RESERVED:
            if x in s:
                return False
        return True

    def _draw(self, fn):
        for _ in range(100):
            s = fn()
            if self._ok(s):
                return s
        raise RuntimeError("could not draw a non-reserved fact")

    def inv(self):
        return self._draw(lambda: f"INV-{self.r.randint(5000, 9899)}")

    def permit(self):
        return self._draw(lambda: f"{self.r.choice(['SP', 'BP', 'EP', 'MP'])}-{self.r.randint(1000, 9999)}")

    def ticket(self):
        return self._draw(lambda: f"{self.r.choice(['KST', 'JNP', 'MRL', 'QLN'])}-{self.r.randint(100, 999)}")

    def price(self):
        def f():
            r = self.r
            n = r.choice([r.randint(40, 900), r.randint(900, 9900), r.randint(10, 300) * 10])
            s = f"${n:,}"
            if r.random() < 0.25:
                s += f".{r.randint(0, 99):02d}"
            return s
        return self._draw(f)

    def port(self):
        return self._draw(lambda: f"port {self.r.randint(3000, 9999)}")

    def version(self):
        def f():
            r = self.r
            a, b, c = r.randint(0, 6), r.randint(0, 14), r.randint(0, 9)
            return r.choice([f"{a}.{b}.{c}", f"v{a}.{b}", f"{a}.{b}.{c}-rc{r.randint(1, 5)}"])
        return self._draw(f)

    def date(self):
        def f():
            r = self.r
            m, d = r.choice(MONTHS), r.randint(1, 28)
            return r.choice([f"{m} {d}", f"the {ordinal(d)} of {m}", f"{r.choice(WEEKDAYS)} {m} {d}"])
        return self._draw(f)

    def supplier(self):
        def f():
            r = self.r
            if r.random() < 0.6:
                return r.choice(SUPPLIERS)
            return f"{r.choice(LAST)} {r.choice(SUPPLIER_SUFFIX)}"
        return self._draw(f)

    def name(self):
        return self._draw(lambda: f"{self.r.choice(FIRST)} {self.r.choice(LAST)}")


class Slots(dict):
    """Template fill-in values. Stable ones (proj, client, person) are set once per note;
    facts (price, inv, port...) are drawn fresh on every use via __missing__."""

    def __init__(self, r: random.Random, facts: Facts, proj: str | None = None, link_fn=None):
        super().__init__()
        self.r, self.f, self.link_fn = r, facts, link_fn
        proj = proj or r.choice(WORK_PROJECTS)
        self["proj"] = proj
        self["client"] = PROJECTS[proj][0]
        self["person"] = facts.name()
        self["person2"] = facts.name()
        self["book"] = f"The {r.choice(BOOK_ADJ)} {r.choice(BOOK_NOUN)}"
        self["recipe"] = r.choice(RECIPES)

    def __missing__(self, k):
        r, f = self.r, self.f
        table = {
            "price": f.price, "price2": f.price, "inv": f.inv, "inv2": f.inv, "port": f.port,
            "version": f.version, "permit": f.permit, "ticket": f.ticket, "date": f.date,
            "supplier": f.supplier,
            "deliverable": lambda: r.choice(DELIVERABLES), "task": lambda: r.choice(TASKS),
            "thing": lambda: r.choice(THINGS), "tool": lambda: r.choice(TOOLS),
            "n": lambda: str(r.randint(2, 12)), "n2": lambda: str(r.randint(3, 9)),
            "bd": lambda: str(r.choice([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15])),
            "link": lambda: self.link_fn() if self.link_fn else "[[Home]]",
        }
        if k not in table:
            raise KeyError(k)
        return table[k]()


# --------------------------------------------------------------------------------------
# Plan: which notes exist (paths) before any text is written, so links can resolve
# --------------------------------------------------------------------------------------


@dataclass
class Item:
    path: str
    kind: str
    opts: dict = field(default_factory=dict)
    proj: str | None = None
    date: datetime | None = None  # date encoded in the file name, if any

    @property
    def title(self) -> str:
        return Path(self.path).stem


def forced_items() -> list[Item]:
    """Structural notes that exercise odd shapes. Truncated at tiny N."""
    return [
        Item("Projects/Juniper/overview.md", "project", {"overview": True}, "Juniper"),
        Item("Projects/Marlow/overview.md", "project", {"overview": True}, "Marlow"),
        Item("Home.md", "root", {"home": True}),
        Item("Studio dashboard.md", "root", {"dashboard": True}),
        Item("Inbox/Parking spot idea.md", "inbox", {"tiny": True}),
        Item("Ideas/Studio cat name.md", "idea", {"tiny": True}),
        Item("Reading/日本語 typography notes.md", "reading", {"unicode": True}),
        Item("Projects/Quillon/series-production-retrospective.md", "project", {"long": True}, "Quillon"),
        Item("Reference/Studio operations handbook.md", "reference", {"long": True, "sop": True}),
        Item("Meetings/2026-06-03 Bramble brand refresh review.md", "meeting", {"dup": True}, "Bramble",
             datetime(2026, 6, 3, tzinfo=timezone.utc)),
        Item("Projects/Bramble/decision-log.md", "project", {"dup": True}, "Bramble"),
        Item("Reading list.md", "root", {"readlist": True}),
        Item("Clients/Juniper Veterinary.md", "client", {}, "Juniper"),
        Item("Clients/Marlow Tea.md", "client", {}, "Marlow"),
        Item("Clients/Quillon Press.md", "client", {}, "Quillon"),
        Item("Clients/Bramble Cycles.md", "client", {}, "Bramble"),
        Item("Clients/Northgate Credit Union.md", "client", {}, "Kestrel"),
    ]


KIND_WEIGHTS = [("journal", 30), ("meeting", 22), ("project", 14), ("reference", 10), ("reading", 6),
                ("idea", 7), ("inbox", 6), ("client", 3)]


def day_back(r: random.Random) -> int:
    """Days before the anchor, skewed so recent months are denser (about 49% in the last 4 months)."""
    return int(HORIZON_DAYS * r.random() ** 1.8)


def at_day(back: int, r: random.Random) -> datetime:
    d = ANCHOR - timedelta(days=back)
    return d.replace(hour=r.randint(7, 22), minute=r.randint(0, 59), second=r.randint(0, 59))


def plan_items(n_filler: int, r: random.Random, used: set[str]) -> list[Item]:
    items: list[Item] = []
    for it in forced_items()[:n_filler]:
        if it.path.lower() not in used:
            used.add(it.path.lower())
            items.append(it)
    n_rest = n_filler - len(items)
    if n_rest <= 0:
        return items

    kinds = r.choices([k for k, _ in KIND_WEIGHTS], [w for _, w in KIND_WEIGHTS], k=n_rest)
    n_long_extra = n_rest // 300
    journal_days: set[int] = set()
    facts = Facts(r)

    def unique(path: str) -> str:
        """Obsidian-style de-dupe: 'name.md', 'name 2.md', 'name 3.md' (case-insensitive)."""
        if path.lower() not in used:
            used.add(path.lower())
            return path
        stem, ext = path[:-3], path[-3:]
        k = 2
        while f"{stem} {k}{ext}".lower() in used:
            k += 1
        p = f"{stem} {k}{ext}"
        used.add(p.lower())
        return p

    for kind in kinds:
        if kind == "journal":
            d = day_back(r)
            while d in journal_days:  # linear probe to the next free day
                d += 1
            journal_days.add(d)
            dt = ANCHOR - timedelta(days=d)
            items.append(Item(unique(f"Journal/{dt:%Y-%m-%d}.md"), "journal", date=dt))
        elif kind == "meeting":
            dt = ANCHOR - timedelta(days=day_back(r))
            proj = r.choice(WORK_PROJECTS)
            title = f"{proj} {r.choice(MEETING_KINDS)}" if r.random() < 0.8 else f"Studio {r.choice(MEETING_KINDS)}"
            items.append(Item(unique(f"Meetings/{dt:%Y-%m-%d} {title}.md"), "meeting", proj=proj, date=dt))
        elif kind == "project":
            proj = r.choice(list(PROJECTS))
            topics = HOUSE_TOPICS if proj == "House" else PROJECT_TOPICS
            opts = {"long": True} if n_long_extra and r.random() < n_long_extra / max(1, n_rest * 0.14) else {}
            items.append(Item(unique(f"Projects/{proj}/{r.choice(topics)}.md"), "project", opts, proj))
        elif kind == "reference":
            x = r.random()
            if x < 0.45:
                title = f"How to {r.choice(HOWTO_VERB).lower()} {r.choice(HOWTO_OBJ)}"
                opts = {"howto": True}
            elif x < 0.65:
                title = f"SOP - {r.choice(SOP_NAMES)}"
                opts = {"sop": True}
            elif x < 0.85:
                title = f"Recipe - {r.choice(RECIPES).title()}"
                opts = {"recipe": True}
            else:
                title = f"{r.choice(TOOLS)} {r.choice(['shortcuts', 'settings', 'plugins', 'export presets'])}"
                opts = {"howto": True}
            items.append(Item(unique(f"Reference/{title}.md"), "reference", opts))
        elif kind == "reading":
            if r.random() < 0.7:
                title = f"The {r.choice(BOOK_ADJ)} {r.choice(BOOK_NOUN)} - {r.choice(AUTHORS)}"
                opts = {"book": True}
            else:
                title = f"Article - {r.choice(ARTICLE_HEADS)}"
                opts = {"article": True}
            items.append(Item(unique(f"Reading/{title}.md"), "reading", opts))
        elif kind == "idea":
            items.append(Item(unique(f"Ideas/{r.choice(IDEA_A)} {r.choice(IDEA_B)}.md"), "idea"))
        elif kind == "inbox":
            t = r.choice(INBOX_TEMPLATES).format_map(Slots(r, facts))
            items.append(Item(unique(f"Inbox/{t}.md"), "inbox"))
        else:  # client
            client = r.choice(CLIENTS_BASE)
            topic = r.choice(["contact log", "proposal", "billing notes", "call notes", "brief"])
            items.append(Item(unique(f"Clients/{client}/{topic}.md"), "client",
                              proj=next((p for p, v in PROJECTS.items() if v[0] == client), None)))
    return items


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


class Renderer:
    def __init__(self, r: random.Random, link_targets: list[tuple[str, str]]):
        self.r = r
        self.facts = Facts(r)
        self.targets = link_targets
        self.cur: Item | None = None

    # ---- small building blocks ----
    def link(self) -> str:
        r = self.r
        title, path = r.choice(self.targets)
        if self.cur is not None and path == self.cur.path:
            title, path = r.choice(self.targets)
        x = r.random()
        if x < 0.62:
            return f"[[{title}]]"
        if x < 0.88:
            alias = r.choice(["the earlier note", "that write-up", title.split()[0].lower(), "my notes on this"])
            return f"[[{title}|{alias}]]"
        return f"[{title}](<{path}>)" if " " in path else f"[{title}]({path})"

    def slots(self, proj: str | None = None) -> Slots:
        return Slots(self.r, self.facts, proj, self.link)

    def para(self, s: Slots, k: int | None = None, pool=None) -> str:
        r = self.r
        pool = pool or WORK_SENT
        sents = r.sample(pool, k or r.randint(2, 4))
        txt = " ".join(x.format_map(s) for x in sents)
        if r.random() < 0.22:
            txt += " " + r.choice(LINK_SENT).format_map(s)
        return txt

    def heading(self, text: str, level: int | None = None) -> str:
        return f"{'#' * (level or self.r.choice([2, 2, 2, 3]))} {text}"

    def callout(self, s: Slots) -> str:
        kind = self.r.choice(["note", "tip", "warning", "info", "example"])
        body = self.para(s, 1)
        return f"> [!{kind}]\n> {body}"

    def table(self, s: Slots) -> str:
        r = self.r
        rows = [f"| {r.choice(DELIVERABLES).capitalize()} | {s['supplier']} | {s['price']} |" for _ in range(r.randint(2, 5))]
        return "| Item | Supplier | Cost |\n|---|---|---|\n" + "\n".join(rows)

    def tasks(self, s: Slots, k: int | None = None) -> str:
        r = self.r
        acts = r.sample(ACTIONS, k or r.randint(2, 4))
        return "\n".join(f"- [{r.choice([' ', ' ', 'x'])}] {a.format_map(s)}" for a in acts)

    def code(self, s: Slots) -> str:
        lang, body = self.r.choice(SNIPPETS)
        body = body.replace("%VERSION%", s["version"]).replace("%PORTNUM%", s["port"].split()[1])
        return f"```{lang}\n{body}\n```"

    def tagline(self, kind: str) -> str:
        r = self.r
        k = r.randint(1, 3)
        return " ".join("#" + t for t in r.sample(TAGS_BY_KIND.get(kind, ["note"]), min(k, len(TAGS_BY_KIND.get(kind, ["note"])))))

    def frontmatter(self, it: Item, created: datetime) -> str:
        r = self.r
        pool = TAGS_BY_KIND.get(it.kind, ["note"])
        tags = r.sample(pool, r.randint(1, min(3, len(pool))))
        if it.proj:
            tags.append(it.proj.lower())
        lines = ["---"]
        if r.random() < 0.5:
            lines.append("tags: [" + ", ".join(tags) + "]")
        else:
            lines.append("tags:")
            lines += [f"  - {t}" for t in tags]
        if r.random() < 0.22:
            lines.append("aliases: [" + ", ".join(json.dumps(a, ensure_ascii=False) for a in (it.title.lower(), it.title.split()[0])) + "]")
        lines.append(f"created: {created:%Y-%m-%d}")
        extras = [
            lambda: f"status: {r.choice(['draft', 'active', 'done', 'waiting', 'parked'])}",
            lambda: f"priority: {r.randint(1, 3)}",
            lambda: f"source: {r.choice(['call', 'email', 'book', 'conversation', 'site visit'])}",
            lambda: f"rating: {r.randint(1, 5)}",
            lambda: f"reviewed: {r.choice(['true', 'false'])}",
            lambda: f"area: {r.choice(AREAS)}",
            lambda: f"project: {it.proj or r.choice(list(PROJECTS))}",
        ]
        for fn in r.sample(extras, r.choice([0, 0, 1, 1, 2, 3])):
            lines.append(fn())
        lines.append("---")
        return "\n".join(lines)

    # ---- per-kind bodies ----
    def render(self, it: Item, created: datetime) -> str:
        self.cur = it
        r = self.r
        body = getattr(self, "body_" + it.kind)(it)
        fm_p = {"journal": 0.45, "meeting": 0.7, "project": 0.7, "reference": 0.85, "reading": 0.8,
                "idea": 0.5, "inbox": 0.3, "client": 0.9, "root": 0.5}[it.kind]
        if it.opts.get("tiny"):
            fm_p = 0.0
        out = body
        if r.random() < fm_p:
            out = self.frontmatter(it, created) + "\n" + body
        return out.rstrip("\n") + "\n"

    def body_journal(self, it: Item) -> str:
        r = self.r
        s = self.slots()
        d = it.date or ANCHOR
        style = r.randint(0, 2)
        parts = []
        if style:
            parts.append(f"# {WEEKDAYS[d.weekday()]} {d.day} {MONTHS[d.month - 1]}")
        work = [self.para(s) for _ in range(r.randint(1, 2))]
        personal = self.para(s, r.randint(1, 2), JOURNAL_PERSONAL)
        if style == 1:
            parts += ["## Work", *work, "## Personal", personal]
        else:
            parts += [*work, personal]
        if r.random() < 0.35:
            parts.append(self.tasks(s, 2))
        if r.random() < 0.3:
            parts.append(self.tagline("journal"))
        return "\n\n".join(parts)

    def body_meeting(self, it: Item) -> str:
        r = self.r
        s = self.slots(it.proj)
        title = it.title.split(" ", 1)[1]
        heads = r.random() < 0.9
        dup = it.opts.get("dup")
        p = ["# " + title, f"**Attendees:** Maren, {s['person']}, {s['person2']}"]
        agenda = "\n".join("- " + a.format_map(s) for a in r.sample(AGENDA, r.randint(2, 4)))
        notes = self.para(s, r.randint(2, 4), MEETING_LINES + WORK_SENT[:6])
        budget = r.choice(MEETING_BUDGET).format_map(s) if r.random() < 0.88 else ""
        if heads:
            p += ["## Agenda", agenda, "## Notes", notes]
            if budget:
                p += [budget]
            p += ["## Decisions", "\n".join("- " + x.format_map(s) for x in r.sample(MEETING_LINES, 2))]
            if dup:  # duplicate headings on purpose
                p += ["## Notes", self.para(s, 2, MEETING_LINES), "## Decisions", "- " + r.choice(MEETING_LINES).format_map(s)]
            p += ["## Action items", self.tasks(s)]
        else:
            p += [notes] + ([budget] if budget else []) + [self.tasks(s, 2)]
        if r.random() < 0.15:
            p.append(self.table(s))
        if r.random() < 0.2:
            p.append(self.tagline("meeting"))
        return "\n\n".join(p)

    def body_project(self, it: Item) -> str:
        r = self.r
        s = self.slots(it.proj if it.proj != "House" else None)
        if it.opts.get("overview"):
            client, what = PROJECTS[it.proj]
            p = [f"# {it.proj} overview",
                 f"{it.proj} is the {what} for {client}. This note is the entry point; everything else lives next to it.",
                 "## Scope", self.para(s, 3), "## People", f"- {s['person']} ({client})\n- {s['person2']} (decisions)",
                 "## Links", f"- {self.link()}\n- {self.link()}"]
            return "\n\n".join(p)
        if it.opts.get("long"):
            return self.long_body(it, s)
        topic = it.title.replace("-", " ")
        p = [f"# {it.proj} {topic}".strip()]
        if it.opts.get("dup"):
            for h in ["Status", "Notes", "Status", "Notes", "Next steps", "Next steps"]:
                p += [f"## {h}", self.para(s, 2)]
            return "\n\n".join(p)
        n_sections = 1 + int(r.lognormvariate(0.4, 0.7))
        headed = r.random() < 0.78
        for i in range(n_sections):
            if headed:
                p.append(self.heading(r.choice(LONG_HEADS), 2 if i % 3 != 2 else 3))
            p.append(self.para(s))
            x = r.random()
            if x < 0.14:
                p.append(self.table(s))
            elif x < 0.24:
                p.append(self.callout(s))
            elif x < 0.34:
                p.append(self.tasks(s))
            elif x < 0.40:
                p.append(self.code(s))
        if r.random() < 0.3:
            p.append(self.tagline("project"))
        if not headed:
            p[0] = p[0] if r.random() < 0.5 else ""
        return "\n\n".join(x for x in p if x)

    def long_body(self, it: Item, s: Slots) -> str:
        """1500+ words, many sections."""
        r = self.r
        p = [f"# {it.title.replace('-', ' ').capitalize()}", self.para(s, 3)]
        heads = list(LONG_HEADS)
        r.shuffle(heads)
        words = 0
        i = 0
        while words < 1650:
            h = heads[i % len(heads)]
            sec = [self.heading(h, 2 if i % 4 != 3 else 3)]
            for _ in range(r.randint(2, 3)):
                sec.append(self.para(s, r.randint(3, 5)))
            x = r.random()
            if x < 0.2:
                sec.append(self.table(s))
            elif x < 0.35:
                sec.append(self.callout(s))
            elif x < 0.5:
                sec.append(self.tasks(s))
            elif x < 0.58:
                sec.append(self.code(s))
            p += sec
            words = sum(len(x.split()) for x in p)
            i += 1
        return "\n\n".join(p)

    def body_reference(self, it: Item) -> str:
        r = self.r
        s = self.slots()
        t = it.title
        if it.opts.get("long"):
            return self.long_body(it, s)
        if it.opts.get("recipe"):
            ing = r.sample(["2 cups flour", "1 tsp salt", "3 cloves garlic", "400 g tomatoes", "1 onion", "olive oil",
                            "250 g lentils", "a handful of parsley", "2 eggs", "100 g butter", "1 lemon", "cumin"], 6)
            p = [f"# {t.replace('Recipe - ', '')}", "## Ingredients", "\n".join("- " + i for i in ing),
                 "## Method",
                 f"1. Prep everything first.\n2. Cook for {r.randint(12, 45)} minutes at {r.choice([160, 180, 200, 220])}C, stirring when it needs it.\n3. Rest for {r.randint(5, 20)} minutes before serving.",
                 r.choice(["Needs more salt than you think.", "Even better the next day.", "Halve the chili for guests.", "Don't skip the lemon at the end."])]
            return "\n\n".join(p)
        p = [f"# {t}"]
        intro = r.choice([
            "Steps I follow so I don't have to remember them.",
            "Written after getting this wrong twice.",
            "Checklist for the studio, kept short on purpose.",
            "Quick reference, last checked against {tool} {version}."]).format_map(s)
        p.append(intro)
        steps = r.sample(STEPS, r.randint(4, 8))
        p.append("\n".join(f"{i + 1}. {x.format_map(s)}" for i, x in enumerate(steps)))
        if r.random() < 0.45:
            p.append(self.code(s))
        if r.random() < 0.4:
            p.append(self.callout(s))
        if r.random() < 0.25:
            p.append(self.table(s))
        if r.random() < 0.3:
            p.append("Related: " + self.link())
        if r.random() < 0.25:
            p.append(self.tagline("reference"))
        if r.random() < 0.18:  # no headings at all
            p[0] = ""
        return "\n\n".join(x for x in p if x)

    def body_reading(self, it: Item) -> str:
        r = self.r
        s = self.slots()
        if it.opts.get("unicode"):
            return ("# 日本語 typography notes\n\nNotes from the Tokyo studio visit — café across the street, naïve first impressions of the kana spacing.\n\n"
                    "- 縦書き (vertical setting) changes where the punctuation sits ✓\n- Line-height for Japanese wants ½ more than Latin\n- Ask Mélanie about the Mincho family → licensing\n\n"
                    "> “The grid is a promise to the reader.”\n\n" + self.para(s, 2))
        ins = r.sample(INSIGHTS, r.randint(2, 5))
        title = it.title
        p = [f"# {title}"]
        if it.opts.get("book"):
            p.append(f"Read in {MONTHS[r.randint(0, 11)]}. Rating {r.randint(2, 5)}/5. About {r.randint(140, 420)} pages.")
        else:
            p.append(f"Saved from a newsletter, about a {r.randint(4, 15)} minute read.")
        p.append("## Highlights")
        p.append("\n\n".join(f"> {x}" + (f" (p. {r.randint(10, 380)})" if r.random() < 0.5 else "") for x in ins[:2]))
        p.append("## Takeaways")
        p.append("\n".join("- " + x for x in ins))
        if r.random() < 0.5:
            p.append(self.para(s, 2) + " " + r.choice(LINK_SENT).format_map(s))
        if r.random() < 0.3:
            p.append(self.tagline("reading"))
        if r.random() < 0.2:
            p = [p[0]] + [x for x in p[1:] if not x.startswith("##")]
        return "\n\n".join(p)

    def body_idea(self, it: Item) -> str:
        r = self.r
        s = self.slots()
        if it.opts.get("tiny"):
            return "Name the studio cat. Shortlist: Kerning, Leading, Gutter."
        txt = f"{it.title}: {r.choice(IDEA_LINES).format_map(s)} {r.choice(IDEA_LINES).format_map(s)}"
        if r.random() < 0.5:
            txt = f"# {it.title}\n\n" + txt
        if r.random() < 0.3:
            txt += "\n\n- [ ] sketch it\n- [ ] ask around"
        if r.random() < 0.3:
            txt += "\n\n" + self.tagline("idea")
        return txt

    def body_inbox(self, it: Item) -> str:
        r = self.r
        s = self.slots()
        if it.opts.get("tiny"):
            return "Ask the council about the loading bay parking spot."
        k = r.random()
        if k < 0.4:
            return f"{it.title}. {r.choice(WORK_SENT).format_map(s)}"
        if k < 0.7:
            return f"- [ ] {it.title}\n- [ ] {r.choice(ACTIONS).format_map(s)}"
        return f"{it.title}\n\n{self.para(s, 1)} {self.link()}"

    def body_client(self, it: Item) -> str:
        r = self.r
        s = self.slots(it.proj)
        client = s["client"]
        rows = "\n".join(f"| {s['inv']} | {r.choice(DELIVERABLES)} | {s['price']} | {r.choice(['paid', 'sent', 'paid'])} |"
                         for _ in range(r.randint(2, 4)))
        p = [f"# {it.title}", f"{client}: {r.choice(['small team', 'family business', 'two-person shop', 'regional chain'])}. Main contact is {s['person']}.",
             "## Billing", f"Terms net {r.choice([14, 30, 45])}. Rate quoted at {s['price']} per day or fixed by milestone.",
             "## History", "| Invoice | Work | Amount | Status |\n|---|---|---|---|\n" + rows,
             self.para(s, 2)]
        return "\n\n".join(p)

    def body_root(self, it: Item) -> str:
        r = self.r
        s = self.slots()
        o = it.opts
        if o.get("home"):
            return ("# Home\n\nStart here. Folders: Projects, Clients, Meetings, Journal, Reference, Reading, Ideas, Inbox.\n\n"
                    "## Active\n- [[Projects/Harbor/permit-timeline|Harbor]]\n- [[Projects/Kestrel/api-notes|Kestrel]]\n- [[Projects/Marlow/print-spec|Marlow]]\n\n"
                    "## Admin\n- [[Client invoice SOP]]\n- [[Suppliers]]\n- [[Backup routine]]\n\n"
                    "> [!tip]\n> Capture in Inbox, file on Fridays.\n")
        if o.get("dashboard"):
            return ("# Studio dashboard\n\n## This week\n- [ ] Invoices out\n- [ ] Clear Inbox\n- [x] Back up the studio drive\n\n"
                    "```dataview\nTABLE status FROM \"Projects\" WHERE status != \"done\"\n```\n\n" + self.para(s, 2))
        if o.get("readlist"):
            titles = ["The Quiet Grid", "Atomic Interfaces", "Slow Typography", "Contracts for Creatives"]
            return "# Reading list\n\n" + "\n".join(f"- [ ] {t}" for t in titles) + "\n\n" + self.para(s, 1, INSIGHTS)
        return "# Note\n\n" + self.para(s)

    def render_tiny_check(self):  # pragma: no cover
        pass


# --------------------------------------------------------------------------------------
# Writing the tree
# --------------------------------------------------------------------------------------


def stamp(dt: datetime) -> float:
    return dt.astimezone(timezone.utc).timestamp()


def write(root: Path, rel: str, data: bytes, mtime: float | None, made: set[Path]) -> None:
    p = root / rel
    if p.parent not in made:
        p.parent.mkdir(parents=True, exist_ok=True)
        made.add(p.parent)
    p.write_bytes(data)
    if mtime is not None:
        os.utime(p, (mtime, mtime))


def nearmiss_doc() -> dict:
    return {"version": 1, "cases": [
        {"id": f"n{i + 1:02d}", "question": q, "gold": [], "expect_none": True, "kind": "nearmiss"}
        for i, q in enumerate(NEARMISS)]}


def generate(n: int, seed: int, out: Path) -> dict:
    if n < MIN_NOTES:
        raise SystemExit(f"--notes must be at least {MIN_NOTES}")
    r = random.Random(seed)
    vault = out / "vault"
    if vault.exists():
        shutil.rmtree(vault)
    out.mkdir(parents=True, exist_ok=True)
    vault.mkdir()
    made: set[Path] = set()

    global RESERVED
    big = n >= SMALL_N_LIMIT
    RESERVED = sorted(set(RESERVED_BASE + (RESERVED_EXT if big else [])))
    plants = PLANTS + (PLANTS_EXT if big else [])
    used = {p.lower() for p, _, _ in plants}
    n_filler = max(0, n - len(plants))
    items = plan_items(n_filler, r, used)

    targets = [(Path(p).stem, p) for p, _, _ in plants] + [(it.title, it.path) for it in items]
    targets = [t for t in targets if not any(x in t[0] for x in RESERVED)]
    rend = Renderer(r, targets)

    for rel, when, text in plants[:n] if n < len(plants) else plants:
        dt = datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        write(vault, rel, text.encode("utf-8"), stamp(dt), made)

    for it in items:
        if it.date is not None and 0 <= (ANCHOR - it.date).days < HORIZON_DAYS:
            mt = at_day((ANCHOR - it.date).days, r)
        else:
            mt = at_day(day_back(r), r)
        text = rend.render(it, mt)
        write(vault, it.path, text.encode("utf-8"), stamp(mt), made)

    # Hidden dirs and non-markdown files: indexers must skip or ignore these.
    write(vault, ".obsidian/app.json", json.dumps(
        {"alwaysUpdateLinks": True, "newFileLocation": "folder", "newFileFolderPath": "Inbox",
         "attachmentFolderPath": "attachments"}, indent=2).encode() + b"\n", stamp(ANCHOR - timedelta(days=300)), made)
    for name, text in TRASH.items():
        write(vault, f".trash/{name}", text.encode("utf-8"), stamp(ANCHOR - timedelta(days=40)), made)
    write(vault, "attachments/logo-sketch.png", PNG_1X1, stamp(ANCHOR - timedelta(days=90)), made)
    write(vault, "attachments/site-visit-notes.txt",
          b"Measured the storefront: 2400mm wide, bracket height 2750mm. Photos in the phone.\n",
          stamp(ANCHOR - timedelta(days=120)), made)

    golden = build_golden(n)
    (out / "golden.json").write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.utime(out / "golden.json", (stamp(ANCHOR), stamp(ANCHOR)))  # fixed mtime keeps the tree byte-stable
    (out / "nearmiss.json").write_text(json.dumps(nearmiss_doc(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.utime(out / "nearmiss.json", (stamp(ANCHOR), stamp(ANCHOR)))

    # Fail loudly if a filler note accidentally contains a reserved (plant-only) fact.
    plant_paths = {p for p, _, _ in plants}
    for it in items:
        txt = (vault / it.path).read_text(encoding="utf-8")
        for res in RESERVED:
            if res in txt and it.path not in plant_paths:
                raise RuntimeError(f"reserved string {res!r} leaked into filler note {it.path}")
    return golden


def md_files(vault: Path):
    """Non-hidden markdown files, vault-relative POSIX paths."""
    for p in sorted(vault.rglob("*.md")):
        rel = p.relative_to(vault)
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield rel.as_posix(), p


def stats(vault: Path) -> dict:
    n = words = chars = 0
    for _, p in md_files(vault):
        t = p.read_text(encoding="utf-8")
        n += 1
        words += len(t.split())
        chars += len(t)
    return {"notes": n, "words": words, "chars": chars, "tokens_est": chars // 4}


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_dir():
            h.update(b"D:" + rel.encode() + b"\n")
        else:
            h.update(b"F:" + rel.encode() + b"\0" + p.read_bytes() + b"\0" + str(int(p.stat().st_mtime)).encode() + b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------


STOPWORDS = frozenset("""
about above after again against all also although always among and another any anyone anything are around
because been before being below between both but can could did does doing done down during each either else
enough even ever every everyone everything for from further get gets getting give gone got had has have having
her here hers him his how into its itself just keep kept let like likely made make makes many may maybe might
more most much must myself near need needs never new next nor not now off often once one only onto other
others our ours out over own per quite rather really same say see seem seems several she should since some
someone something still such take taken tell than that the their theirs them then there these they thing
things think this those though through thus too took toward towards under until upon use used uses using very
want wants was way ways well went were what when whenever where whether which while who whom whose why will
with within without would yet you your yours
don isn aren doesn didn won wasn weren hasn haven wouldn couldn shouldn
""".split())
PARAPHRASE_DF_LIMIT = 0.03  # words found in more than 3% of the vault's notes are not distinctive
PREFIX = 5


def tokens(text: str) -> set[str]:
    """Lowercase words of 3+ letters, stopwords removed."""
    return {w for w in re.findall(r"[^\W\d_]{3,}", text.lower()) if w not in STOPWORDS}


def doc_freq(texts) -> tuple[dict[str, int], int]:
    df: dict[str, int] = {}
    n = 0
    for t in texts:
        n += 1
        for w in tokens(t):
            df[w] = df.get(w, 0) + 1
    return df, n


def paraphrase_overlap(question: str, note_text: str, df: dict[str, int], n_notes: int) -> list[str]:
    """Distinctive question tokens that share a 5-letter prefix with a distinctive token of the note."""
    def keep(ts):
        return {w for w in ts if df.get(w, 0) <= PARAPHRASE_DF_LIMIT * n_notes}
    note_pref = {w[:PREFIX] for w in keep(tokens(note_text))}
    return sorted(w for w in keep(tokens(question)) if w[:PREFIX] in note_pref)


def self_check(n: int, seed: int) -> int:
    problems: list[str] = []
    notes: list[str] = []

    def check(ok: bool, msg: str):
        (notes if ok else problems).append(("ok   " if ok else "FAIL ") + msg)

    tmp = Path(tempfile.mkdtemp(prefix="vaultcheck-"))
    try:
        # 1. determinism, across processes with different hash seeds
        hashes = []
        t0 = time.time()
        for i, hs in enumerate(("1", "2")):
            d = tmp / f"run{i}"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--notes", str(n), "--seed", str(seed),
                            "--out", str(d), "--quiet"], check=True, env={**os.environ, "PYTHONHASHSEED": hs})
            hashes.append(tree_hash(d))
        check(hashes[0] == hashes[1], f"determinism: two runs give the same tree hash ({hashes[0][:12]})")
        out = tmp / "run0"
        vault = out / "vault"
        golden = json.loads((out / "golden.json").read_text(encoding="utf-8"))
        cases = golden["cases"]

        # 2. N respected, hidden dirs present
        paths = dict(md_files(vault))
        check(len(paths) == n, f"non-hidden .md count == N ({len(paths)} vs {n})")
        check((vault / ".obsidian/app.json").is_file(), ".obsidian/app.json exists")
        check(len(list((vault / ".trash").glob("*.md"))) == 2, ".trash holds 2 notes")
        check(any((vault / "attachments").iterdir()), "attachments/ holds files")
        text = {rel: p.read_text(encoding="utf-8") for rel, p in paths.items()}
        all_text = "\n".join(text.values())
        hidden_text = "\n".join(p.read_text(encoding="utf-8") for p in (vault / ".trash").glob("*.md"))
        for d in DECOY_STRINGS:
            check(d not in all_text and d in hidden_text, f"trash decoy {d!r} only in .trash")

        # 3. golden schema + paths + substrings
        check(golden.get("version") == 1, "golden version == 1")
        kinds: dict[str, int] = {}
        for c in cases:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
            check(set(c) >= {"id", "question", "gold", "expect_none", "kind", "answer_contains"}, f"{c['id']} has required keys")
            if c["kind"] == "no_answer":
                check(c["expect_none"] is True and c["gold"] == [], f"{c['id']} no_answer shape")
                continue
            check(all(g in text for g in c["gold"]), f"{c['id']} gold paths exist")
            gold_txt = "\n".join(text[g] for g in c["gold"] if g in text)
            for a in c["answer_contains"]:
                check(a in gold_txt, f"{c['id']} answer {a!r} found in gold")
                others = [rel for rel, t in text.items() if a in t and rel not in c["gold"]]
                check(not others, f"{c['id']} answer {a!r} is unique to gold notes" + (f" (also in {others[:2]})" if others else ""))
            if len(c["gold"]) > 1:
                check(all(any(a in text[g] for a in c["answer_contains"]) for g in c["gold"]),
                      f"{c['id']} each multi-note gold contributes an answer")
            if c["kind"] == "updated":
                for s in c["stale_notes"]:
                    check(s in text, f"{c['id']} stale note exists")
                    check(all(a not in text[s] for a in c["answer_contains"]), f"{c['id']} stale note lacks the new answer")
                    check(os.stat(vault / c["gold"][0]).st_mtime > os.stat(vault / s).st_mtime, f"{c['id']} gold is newer than stale")
                    check("pdate" in text[c["gold"][0]], f"{c['id']} gold says update explicitly")
        answerable = sum(v for k, v in kinds.items() if k != "no_answer")
        need = 8 if n < SMALL_N_LIMIT else 100
        check(answerable >= need, f"{answerable} answerable golden cases (need >= {need})")
        check(len({c["id"] for c in cases}) == len(cases), "golden ids are unique")
        if n >= SMALL_N_LIMIT:
            minimum = {"direct": 8, "paraphrase": 40, "entity": 4, "multi_note": 3, "updated": 2}
            check(all(kinds.get(k, 0) >= m for k, m in minimum.items()) and kinds.get("no_answer") == 5, f"kind counts {kinds}")
            # the original 30 cases are frozen: same ids, same text, in the same order
            check([(c["id"], c["question"], c["gold"]) for c in cases[:30]] ==
                  [(f"g{i + 1:03d}", c["question"], c["gold"]) for i, c in enumerate(CASES)], "original 30 golden cases unchanged")
            # a paraphrase case shares no distinctive word with its gold note (see module docstring)
            df, n_notes = doc_freq(text.values())
            para = [c for c in cases if c["kind"] == "paraphrase"]
            good = 0
            legacy = {f"g{i + 1:03d}" for i in range(len(CASES))}  # frozen cases: reported, never edited
            for c in para:
                shared = paraphrase_overlap(c["question"], "\n".join(text[g] for g in c["gold"]), df, n_notes)
                good += not shared
                if c["id"] in legacy:
                    notes.append(f"info {c['id']} (original case) paraphrase overlap with gold: {shared or 'none'}")
                else:
                    check(not shared, f"{c['id']} paraphrase shares no distinctive word with its gold note" + (f" (shares {shared})" if shared else ""))
            check(good >= 40, f"{good} paraphrase cases pass the no-shared-words check (need >= 40)")
            gold_notes = {g for c in cases for g in c["gold"]}
            check(len(gold_notes) >= 60, f"{len(gold_notes)} distinct gold notes (need >= 60)")

        # 3b. nearmiss.json: unanswerable prompts that reuse the vault's vocabulary
        nm = json.loads((out / "nearmiss.json").read_text(encoding="utf-8"))
        nmc = nm["cases"]
        check(nm.get("version") == 1 and len(nmc) >= 30, f"nearmiss.json: {len(nmc)} prompts (need >= 30)")
        check(len({c["id"] for c in nmc}) == len(nmc) and len({c["question"] for c in nmc}) == len(nmc), "nearmiss ids and prompts are unique")
        check(nm == nearmiss_doc(), "nearmiss.json equals what the generator emits")
        vocab_df, _ = doc_freq(text.values())
        golden_q = {c["question"] for c in cases}
        answers = [a for c in cases for a in c["answer_contains"]]
        low_all = all_text.lower()
        for c in nmc:
            q = c["question"]
            check(set(c) == {"id", "question", "gold", "expect_none", "kind"} and c["gold"] == [] and c["expect_none"] is True
                  and c["kind"] == "nearmiss", f"{c['id']} nearmiss shape")
            check(q not in golden_q, f"{c['id']} is not a golden question")
            check(not any(a.lower() in q.lower() for a in answers), f"{c['id']} contains no golden answer fact")
            ids = re.findall(r"\b[A-Za-z]{2,}-\d[\w-]*|\b\d{3,}\b|\b\d+[.:]\d+\b", q)
            check(all(i.lower() not in low_all for i in ids), f"{c['id']} numbers/identifiers {ids} absent from the vault")
            vocab = [w for w in tokens(q) if len(w) >= 5 and w in vocab_df]
            check(len(vocab) >= 1, f"{c['id']} reuses the vault's vocabulary ({vocab[:4]})")

        # 4. no_answer terms really absent (case-insensitive) from every vault text file
        low = all_text.lower() + hidden_text.lower()
        for q, terms in NO_ANSWER:
            for t in terms:
                check(t.lower() not in low, f"no_answer term {t!r} absent from the vault")

        # 5. mtimes
        ms = [os.stat(vault / rel).st_mtime for rel in paths]
        lo, hi = min(ms), max(ms)
        span = (hi - lo) / 86400
        check(hi <= stamp(ANCHOR + timedelta(days=1)), "no mtime after the anchor date")
        check(lo >= stamp(ANCHOR - timedelta(days=HORIZON_DAYS + 5)), "no mtime older than ~14 months")
        check(span >= 300, f"mtime span {span:.0f} days (>= 300)")
        if n >= SMALL_N_LIMIT:
            recent = sum(1 for m in ms if m >= stamp(ANCHOR - timedelta(days=120)))
            oldest = sum(1 for m in ms if m <= stamp(ANCHOR - timedelta(days=305)))
            check(recent > oldest * 1.5, f"denser in recent months ({recent} in last 4 mo vs {oldest} in first 4 mo)")

        # 6. realism features (meaningful from N=100)
        if n >= SMALL_N_LIMIT:
            fm = sum(1 for t in text.values() if t.startswith("---\n"))
            check(0.5 <= fm / len(text) <= 0.7, f"frontmatter share {fm / len(text):.2f} in [0.5, 0.7]")
            has = lambda f: any(f(t) for t in text.values())  # noqa: E731
            wc = {rel: len(t.split()) for rel, t in text.items()}
            check(has(lambda t: "\n#" not in "\n" + t.split("---\n", 2)[-1] if t.startswith("---\n") else not any(l.startswith("#") and l.lstrip("#").startswith(" ") for l in t.splitlines())),
                  "some notes have no headings")
            check(has(lambda t: any(t.count(h) > 1 for h in ("## Notes", "## Status", "## Decisions"))), "some notes repeat a heading")
            check(max(wc.values()) >= 1500, f"a long note exists ({max(wc.values())} words)")
            check(min(wc.values()) < 40, f"a tiny note exists ({min(wc.values())} words)")
            base: dict[str, int] = {}
            for rel in text:
                b = Path(rel).name
                base[b] = base.get(b, 0) + 1
            check(any(v > 1 for v in base.values()), "two files share a basename")
            check(any("&" in rel and "(" in rel and " " in rel for rel in text), "a title has spaces, parentheses and ampersand")
            check(has(lambda t: "| --- |" in t or "|---|" in t), "tables")
            check(has(lambda t: "```" in t), "code fences")
            check(has(lambda t: "- [ ]" in t), "task lists")
            check(has(lambda t: "> [!" in t), "callouts")
            check(has(lambda t: "[[" in t and "|" in t.split("[[", 1)[1].split("]]", 1)[0]), "[[Note|alias]] links")
            check(has(lambda t: "](" in t and (".md)" in t or ".md>)" in t)), "markdown links")
            check(has(lambda t: "[[" in t), "wikilinks")
            check(has(lambda t: "aliases:" in t), "aliases frontmatter")
            check(has(lambda t: any(ord(ch) > 127 for ch in t)), "unicode content")
            check(any(l.startswith("Journal/") for l in text) and any(l.startswith("Meetings/") for l in text), "journal and meeting notes present")
        check(time.time() - t0 < 120, f"self-check wall time {time.time() - t0:.1f}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for line in notes:
        if "--verbose" in sys.argv:
            print(line)
    for line in problems:
        print(line)
    print(f"self-check N={n}: {len(notes)} passed, {len(problems)} failed")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--notes", type=int, default=100, help="number of non-hidden .md notes (>= 30)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, help="output directory (gets vault/, golden.json and nearmiss.json)")
    ap.add_argument("--self-check", action="store_true", help="generate twice and verify the fixture invariants")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check(a.notes, a.seed)
    if not a.out:
        ap.error("--out is required")
    t0 = time.time()
    golden = generate(a.notes, a.seed, a.out)
    if not a.quiet:
        st = stats(a.out / "vault")
        kinds: dict[str, int] = {}
        for c in golden["cases"]:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        print(f"wrote {st['notes']} notes, {st['words']} words, ~{st['tokens_est']} tokens (chars/4), "
              f"{len(golden['cases'])} golden cases {kinds} in {time.time() - t0:.1f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
