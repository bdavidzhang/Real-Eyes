# Open Reality — Slide Deck Plan
## 3 slides · ~1 minute of talking · 2 minutes of judges using the app

---

## CONTEXT (paste this whole file into Claude web to generate slides)

**Project:** Open Reality — spatial AI platform built at HackIllinois 2026, Modal track
**Format:** 3-minute in-person presentation. Strategy: 1 minute of slides to set the story, 2 minutes of judges exploring the live app themselves.
**Goal:** Make them feel the stakes, understand what it does, and want to touch it.

**What it does:** Point any phone at a room — no app, no hardware — and a live 3D map builds in real time on Modal's H100 GPU. An AI agent finds objects by name, pins them in 3D, and answers spatial questions using real geometry.

---

## DESIGN SYSTEM

**Background:** Deep near-black `#0A0A0F`
**Primary accent:** Electric cyan `#00D4FF`
**Emotional accent:** Warm amber `#FF9A00` — problem slide only
**Text:** White headlines, `#A8B2C0` body
**Font:** Bold geometric sans-serif (Inter or similar)
**Motif:** Sparse particle/point cloud field in backgrounds — suggests 3D space without being busy
**Ratio:** 16:9
**Principle:** One dominant visual per slide. Judges read a slide in 2 seconds — make those 2 seconds count.

---

## SLIDE 1 — THE STORY

**Purpose:** Make the problem visceral before showing anything technical.

**Layout:** Full-bleed. Three human scenarios, vertically stacked, centered. No columns — these should feel like gut punches, one at a time.

**Background:** Near-black with a very faint red tint — urgency without being garish.

**Three lines, large, centered, with slight vertical spacing between each:**
```
A firefighter enters a building they've never seen.
A rescue team navigates rubble without a map.
A blind person walks into a new space alone.
```
Each line in white. Slightly smaller than a traditional headline — the weight comes from the words, not the size.

**Full-width line below, in amber, larger:**
```
AI could help all of them — if it could actually see.
```

**Speaker notes (20 seconds):**
> "Spatial awareness is life-or-death for a lot of people. And right now, AI is completely blind to physical space. It can describe what a photo looks like — but it has no idea where anything actually *is*."

---

## SLIDE 2 — WHAT IT DOES

**Layout:** Left-right split. 55/45.

**Left side (55%):** Large screenshot of the Open Reality 3D viewer — a colorful dense point cloud with glowing 3D bounding boxes labeled "fire extinguisher" and "exit sign." Camera frustums visible. Should look real and impressive.

**Right side (45%):**

Headline, stacked, large to small:
```
One phone.
Any space.
A 3D map AI
can reason about.
```

Three compact rows below (icon + text, no bullet points):
```
☁️  Runs on Modal — H100 GPU, live from a link
🧠  Agent finds objects, answers spatial questions
📱  No app. No hardware. Just a browser.
```

**Speaker notes (25 seconds):**
> "Open Reality. Point any phone at a room. A 1B-parameter vision model running on Modal's H100 builds a live 3D map in real time. An AI agent searches it — finds objects by name, answers spatial questions with real geometry. One command to deploy. Accessible from a link."

---

## SLIDE 3 — TRY IT

**Purpose:** Hand control to the judges. This slide stays up for 2 minutes.

**Layout:** Centered. Minimal. The QR code is the hero.

**Background:** The completed 3D point cloud, slightly blurred, as a full-bleed background. Dark overlay for readability. Beautiful and inviting.

**Center, top:**
```
Open Reality
```
"Open" white, "Reality" cyan. Large.

**Center, below — very large QR code:**
`[ QR CODE — links to live Modal-deployed URL ]`

**Below QR, small white text:**
```
Point your phone at anything in this room.
Watch it map in real time.
```

**Bottom strip, left:**
```
github.com/bdavidzhang/Real-Eyes
```

**Bottom strip, right, cyan:**
```
HackIllinois 2026 · Modal Track
```

**Speaker notes (10 seconds, then go quiet and let them explore):**
> "Scan this. Point your phone at anything in the room. The map will build live — and you can ask the agent to find something."

---

## TIMING

| Segment | Time |
|---------|------|
| Slide 1 — The Story | 0:00 – 0:20 |
| Slide 2 — What It Does | 0:20 – 0:45 |
| Slide 3 — Try It (stays up) | 0:45 – 3:00 |
| Q&A | 3:00 – 5:00 |

---

---

## Q&A BACKUP SLIDES (shown on request during the 2-minute Q&A, not during the main pitch)

---

## SLIDE 4 — UNDER THE HOOD: MODAL INFERENCE

**Purpose:** Satisfy technical judges. Show that this is serious, research-grade engineering running on real infrastructure — not a wrapper.

**Layout:** Two columns. Left: the pipeline diagram. Right: Modal-specific callouts.

**Background:** Dark with a faint cyan grid. Feels precise, technical, credible.

**Top headline:**
```
Real-time spatial AI is a hard inference problem. Modal makes it possible.
```
White. Medium size. Confident.

**Left column (55%) — Pipeline flow diagram, top to bottom:**

Five nodes stacked vertically with labeled arrows between them:

```
[ Phone Camera Frame ]
        ↓  WebSocket stream (HTTPS via Modal tunnel)
[ Keyframe Selection ]
   Lucas-Kanade optical flow — skip frames without motion
        ↓
[ VGGT-1B Vision Model ]
   Predicts dense depth + camera pose per frame
   No GPS. No depth sensor. Just pixels.
        ↓
[ CLIP + SAM3 ]
   CLIP scores every submap against target queries
   SAM3 segments matches → projects to 3D bounding box
        ↓
[ GTSAM Pose Graph ]
   SL(4) manifold optimization
   Loop closure keeps the global map consistent
```

Each node: white label, cyan arrow, small grey descriptor text underneath.

**Right column (45%) — Modal callouts, as a tight stack of cards:**

Each card has a cyan left border, dark background, icon, title, and 1-line detail:

```
⚡ Warm Containers
   No cold starts between frames.
   Inference hits in <100ms per submap.

💾 Modal Volumes
   VGGT-1B (4GB) + DINO-Salad weights
   cached across runs. First inference: seconds.

🔒 Modal Tunnel
   Provides the HTTPS endpoint the phone
   camera requires to stream. Zero config.

🚀 One Command
   modal deploy modal_streaming.py
   Stable public URL. Live for anyone.
```

**Speaker notes (for Q&A, ~40 seconds):**
> "The inference stack is three heavy models in sequence: VGGT-1B for depth and pose, CLIP for open-vocabulary object scoring, SAM3 for segmentation. On a laptop, this pipeline is way too slow for real-time use. On Modal, we keep the container warm between WebSocket frames — no cold start penalty. Model weights live in Modal Volumes so we don't re-download 4GB on every run. And Modal's tunnel is what gives the phone camera the HTTPS endpoint it needs to stream without any SSL setup on our end. The whole thing deploys in one command."

---

## SLIDE 5 — WHERE THIS GOES NEXT

**Purpose:** Show vision. Judges want to know the team thinks beyond the hackathon. This slide earns points for ambition and social impact depth.

**Layout:** Top headline. Then two sections side-by-side: Use Cases (left), Scale Path (right).

**Background:** Dark, warm — slightly different from the technical slide. Feels expansive, forward-looking. Faint amber glow in corners.

**Top headline:**
```
Spatial intelligence for anyone, anywhere.
```
White. Large. The thesis.

**Left section — USE CASES (label: "Who needs this now"):**

Five rows, each a bold label + one-line context. No icons needed — the labels carry the weight:

```
🚒  First Responders
    Pre-scan buildings before entering a fire or active scene.

♿  Accessibility
    Navigation context for the visually impaired in new spaces.

🤖  Robotics & Simulation
    Generate real-world 3D training data from any phone.

🏗️  Construction & Inspection
    Live spatial documentation without LiDAR hardware.

🔍  Disaster Response
    Rapid structural mapping in search-and-rescue operations.
```

**Right section — SCALE PATH (label: "What's next"):**

Three steps, styled like a roadmap — connected by a vertical line with nodes:

```
NOW
Deployed on Modal · H100 GPU · Any phone · One command

↓

NEXT
Multi-user collaborative maps — multiple phones, one shared 3D space.
Edge inference for offline / low-bandwidth environments.

↓

FUTURE
Open API — any developer adds spatial intelligence to their agent.
The spatial layer for the agentic internet.
```

**Bottom strip, full width, cyan text, centered:**
```
The hardware barrier is gone. The deployment barrier is gone.
What remains is the mission.
```

**Speaker notes (for Q&A, ~35 seconds):**
> "The use cases that matter most are the ones where spatial awareness is a safety issue — first responders, accessibility, disaster response. But because it's built on Modal and deployed via a link, any developer can build on top of it. The next step is a public API: give any AI agent a spatial tool it can call. And longer term — multi-user collaborative maps, edge inference for offline scenarios. We think this is the spatial layer that agentic AI has been missing."

---

## PRODUCTION NOTES FOR CLAUDE WEB

Generate as: HTML/CSS single-file presentation, or Reveal.js, or detailed Figma/Google Slides layout — whichever format the user specifies.

- Slide 3 is the most important. It needs to feel open and inviting, not busy.
- The QR code on slide 3 should be large enough to scan from 4–5 feet away.
- Slides 4 and 5 are Q&A backup — they should match the design system but can be slightly denser since judges are leaning in to ask questions.
- No animations except subtle fade between slides. The content is the drama.
- If generating HTML: make slide 3 loop or stay static — it doesn't auto-advance.
