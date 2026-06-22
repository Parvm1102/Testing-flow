# Gridlock — Demo Video Script (3–4 minutes)

> A scene-by-scene shooting script for the demo video. Each scene gives you:
> **[SHOW]** what's on screen · **[SAY]** the voiceover (≈150 wpm) · **[TEXT]** on-screen
> captions/callouts · **[TIME]** running length.
>
> **Total target: ~3:30.** Tighten by trimming Scene 6 (Top Areas) or the Models deep-dive
> if you run long. **Record the screen at 1080p+**, do a dry run of every click first, and
> pre-fill the Predict form so you're never typing on camera.
>
> **Tone:** confident, fast, product-demo energy. Let the app do the talking — narrate the
> *decision*, not the UI mechanics ("now I click this button").

---

## PRE-PRODUCTION CHECKLIST (do this before recording)
- [ ] App running and warm (open it once so the model is loaded — first predict can be slow).
- [ ] Browser in a clean window, no bookmarks bar, 100–110% zoom, dark OS theme.
- [ ] Pre-decide the **demo event** so the result is impressive and consistent:
      `event_cause = construction`, location near **HAL Old Airport Road**, a weekday evening
      time. (Construction reliably yields HIGH manpower + closure-likely + chronic-hotspot
      flag — a strong, story-friendly result.)
- [ ] Have a **second event** ready (e.g. `pot_holes` at J.P. Nagar) if you want a contrast.
- [ ] Map tab pre-loaded once so tiles are cached.
- [ ] Screenshots/figures ready as cutaways: `reports/figures/*.png`, the architecture
      diagram (export the Mermaid from the deck), and a 1-line dataset stat card.
- [ ] Optional: a webcam talking-head bubble in the corner for the intro/outro only.

---

## SCENE 1 — HOOK + PROBLEM  ·  [TIME 0:00–0:25]

**[SHOW]** Cold open on the **dark Map tab** of the live app, slowly zoomed on Bengaluru with
hotspot markers glowing. (Don't show the title card yet — open on the product.)

**[SAY]**
> "Every day, a city traffic-control room faces the same questions when an event is
> reported: do we close this road? Divert traffic? How many officers do we send — and how
> long will it tie up the corridor? Today, those calls are made *reactively*, on
> experience. By the time the response is right-sized, the jam has already spread."

**[TEXT]** (lower third, fades in/out)
- "Bengaluru · 8,000+ road events"
- "The decisions are reactive. The data could make them predictive."

**[TIME]** 25s

---

## SCENE 2 — THE SOLUTION (title beat)  ·  [TIME 0:25–0:45]

**[SHOW]** Quick, clean **title card**: "Gridlock — forecast the impact, pre-position the
response." Then cut to the **Predict tab** empty form.

**[SAY]**
> "This is **Gridlock**. The moment an event is reported, it forecasts four things — and
> turns each one into a concrete action: a manpower level, a closure-and-diversion call, a
> clearance time, and an early warning for spots that keep reoffending. Let me show you."

**[TEXT]**
- "Gridlock"
- "4 forecasts → 4 decisions, in one click"

**[TIME]** 20s

---

## SCENE 3 — LIVE PREDICTION (the core demo)  ·  [TIME 0:45–1:35]

**[SHOW]** The **Predict** form. Pick a **police station** from the dropdown → the map pin and
coordinates **auto-fill** (call this out — it's a nice touch). Set `event_cause = construction`,
event type, a weekday-evening start time, a short description ("Metro construction, lane
blocked"). Click **Forecast**. Let the **result panel** render.

**[SAY]**
> "I only enter what's actually known when an event is reported — the location and start
> time are required; everything else just sharpens the forecast. I'll pick a spot on Old
> Airport Road, mark it as construction, and forecast."
>
> *(result appears)*
>
> "Instantly, Gridlock returns a **decision panel**, not raw numbers. Up top: the
> recommended **manpower level — HIGH — with a suggested officer count**. Then the
> **closure probability** with an expected-closure badge, the **operational priority**, the
> **clearance time with an 80% confidence band**, and a **chronic-hotspot flag** for this
> location. It even spells out the **barricading and diversion** plan and a one-line
> rationale."

**[TEXT]** (callout arrows on the panel as you mention each)
- "Manpower tier + officers"
- "Closure probability (calibrated)"
- "Duration + 80% interval"
- "Chronic-hotspot flag"
- "Recommended action + rationale"

**[TIME]** 50s

> **Tip:** Hover one **(i) info popover** for half a second so viewers see every metric is
> self-explaining. Don't read it aloud — just let it appear.

---

## SCENE 4 — WHY IT'S TRUSTWORTHY (the differentiator)  ·  [TIME 1:35–2:05]

**[SHOW]** Cut to a **half-screen**: keep the result panel on one side; on the other, drop a
simple **"leakage caught" caption card** (a static graphic you make), e.g.:
"Strongest raw predictor scored AP ≈ 0.98 → it was the *answer leaking in* → deleted on
purpose." Optionally flash the architecture diagram briefly.

**[SAY]**
> "Here's what makes this more than a hackathon model. The dataset shipped with **no impact
> label** — we engineered all four targets ourselves. And it was full of **leakage**: the
> single strongest predictor of a closure was a feature that only exists *because* the
> closure already happened. We **deleted our most powerful feature on purpose** — because
> using it would be cheating. Every feature Gridlock uses is **causal**: each event only
> ever sees events reported *before* it."

**[TEXT]**
- "No label existed → we engineered 4 targets"
- "Deleted our strongest feature → it was leakage"
- "Every feature is causal · train on the past, test on the future"

**[TIME]** 30s

---

## SCENE 5 — THE MAP (city-scale intelligence)  ·  [TIME 2:05–2:35]

**[SHOW]** Switch to the **Map tab**. Click through the four views: **Congestion → Chronic
hotspots → Closure risk → Manpower load**. Hover one station marker so the **stat popup**
shows. Linger on **Chronic hotspots** (the ~110 m cells) since it's your unique model.

**[SAY]**
> "Zooming out, the same engine powers a city-wide view. Toggle between **congestion
> density**, **chronic hotspots** — the ~110-metre spots likely to reoffend — **closure
> risk**, and **predicted manpower load**. This is the difference-maker: instead of
> repeatedly patrolling the same junction, the control room sees exactly where to send a
> **permanent fix** — drainage, resurfacing, a marshal."

**[TEXT]**
- "4 city views: Congestion · Hotspots · Closure · Manpower"
- "Chronic hotspots → fix the spot, don't re-patrol it"

**[TIME]** 30s

---

## SCENE 6 — TOP AREAS + MODELS (proof)  ·  [TIME 2:35–3:05]

**[SHOW]** Quick cut to **Top Areas** — click a column header to **sort** (e.g. by risk or
closure rate) so the re-sort animates. Then a fast cut to the **Models** tab: scroll past
the **PR / calibration / SHAP** charts and the **operating-points** table. Keep this snappy —
it's a montage, not a walkthrough.

**[SAY]**
> "Every police-station area is ranked and fully sortable, so commanders can find the worst
> corridors in two clicks. And we don't ask anyone to take our word for it — the **Models**
> tab is the evidence: precision-recall and **calibration** curves, feature importance, and
> the **operating points** that let you dial risk posture up for a VIP day, or down for
> routine ops. Same model, different stance."

**[TEXT]**
- "Every area ranked · fully sortable"
- "Calibrated, explainable, policy-aware"

**[TIME]** 30s

> **If running long:** cut the Top Areas beat and keep only a 6-second Models montage.

---

## SCENE 7 — CLOSE (impact + it's live)  ·  [TIME 3:05–3:30]

**[SHOW]** Pull back to the **Predict result panel** or the dark Map. End card with the
**live URL / QR code** and the tagline.

**[SAY]**
> "Gridlock turns a messy, unlabeled event log into foresight a control room can act on —
> calibrated, honest, and already **deployed live**, running on CPU with no GPU bill. It
> doesn't just predict traffic. It tells a city where to stand **before** the jam forms."

**[TEXT]**
- "Live now → <your Hugging Face Spaces URL>"  (add a QR code)
- "Gridlock · forecast the impact, pre-position the response"

**[TIME]** 25s

---

## FIGURES & CUTAWAYS TO PREPARE (referenced above)
| Cutaway | Where it's used | Source |
|---|---|---|
| Dark Map screen-capture (intro) | Scene 1, 7 | live app, Map tab |
| Title card | Scene 2 | make in deck/Canva |
| Predict result panel (with callout arrows) | Scene 3 | live app + annotate in editor |
| "Leakage caught" caption card | Scene 4 | static graphic (1 line of text) |
| Architecture diagram (brief flash) | Scene 4 | export Mermaid from the deck (mermaid.live) |
| PR / calibration / SHAP plots | Scene 6 | `reports/figures/*.png` |
| End card + QR to live URL | Scene 7 | make in deck/Canva |

---

## TIMING SUMMARY
| Scene | Beat | Length | Cumulative |
|---|---|---|---|
| 1 | Hook + problem | 0:25 | 0:25 |
| 2 | Solution / title | 0:20 | 0:45 |
| 3 | Live prediction | 0:50 | 1:35 |
| 4 | Why it's trustworthy | 0:30 | 2:05 |
| 5 | Map | 0:30 | 2:35 |
| 6 | Top Areas + Models | 0:30 | 3:05 |
| 7 | Close | 0:25 | 3:30 |

**Word budget:** ~520–540 spoken words total at ~150 wpm ≈ 3:30. If you narrate faster
(~165 wpm) you'll land closer to 3:00, leaving room for silent UI beats.

---

## DELIVERY TIPS
- **Pre-fill and rehearse the Predict form** off-camera; on camera just pick the station and
  click Forecast so there's no dead typing time.
- **Cut on motion** — switch scenes right as a panel renders or a sort animates; it keeps pace.
- **Narrate decisions, not clicks.** Say "the recommended manpower is HIGH," never "I click
  the predict button."
- **Captions on:** many judges watch muted first — the **[TEXT]** lines carry the story alone.
- **One impressive result > three mediocre ones.** Lock the construction event so the demo is
  repeatable and always lands HIGH/closure/hotspot.
- **End on the live URL/QR** and hold it for 3 seconds so judges can open it themselves.
