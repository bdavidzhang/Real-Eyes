## Project name
Open Reality

## Elevator pitch

Imagine Open Claw, but for physical reality! Open Reality gives AI agents eyes to see and reason about space. 

No special hardware. No expertise. Just the missing layer between language and the physical world.

---

## Project details

### The Problem

Every year, first responders walk into buildings they've never seen. Search-and-rescue teams navigate rubble without a map. Visually impaired people enter new spaces with no spatial context. Emergency sweeps happen by memory and guesswork.

AI could help — if it could actually see.

Today's AI agents are spatially blind. They process pixels and text, but have no concept of *where* things are in 3D space. Ask a state-of-the-art LLM "how many chairs are in this room?" while pointing a camera at it — and it will hallucinate an answer. It can describe what an object looks like, but not where it lives in space, how far away it is, or what's around the corner.

The tools that do provide spatial maps — LiDAR rigs, depth cameras, survey hardware — cost tens of thousands of dollars and require trained operators. Spatial intelligence has stayed locked inside robotics labs.

Open Reality breaks that barrier.

---

### What We Built

Open Reality is a cloud-native spatial AI platform. Describe your task. Point a phone. An AI agent maps your space in real time, finds your targets, and answers spatial questions — from any browser, from anywhere.

**The experience:**

1. Type your mission in plain English: *"I'm a paramedic doing a safety sweep."*
2. Open Reality generates a spatial plan — what to look for, how to move through the space.
3. Open the camera stream on your phone. No app. Just a link.
4. Walk the space. A 3D map builds live.
5. The agent finds your targets automatically, pins them in 3D, and answers follow-up questions when you're done.

**Where Modal makes this real:**

The inference pipeline running under the hood — a 1-billion-parameter vision model, CLIP scoring on every submap, SAM3 segmentation — is far too heavy for a laptop and too latency-sensitive for a slow API call. Modal runs it on an H100 GPU, kept warm between frames so nothing drops. Modal Volumes persist the model weights so first inference is seconds, not minutes. A Modal tunnel gives the phone camera the secure HTTPS connection it needs to stream. `modal deploy` — and it's live.

This is Modal the way it's meant to be used: serious inference, at real-time speed, accessible from a link.

---

### Why It Matters

The people who need spatial awareness most are the ones who can least afford to wait. A firefighter doesn't have time to set up hardware. A paramedic doing a sweep is already on the clock. A blind person navigating a new building deserves a tool that just works.

Open Reality works with the phone already in your pocket. Deployed in ten seconds. No installation. No expertise. No hardware.

For developers: an open agentic platform. Bring your own queries, extend the agent's tools, connect via WebSocket. The blueprint for spatial + language AI — available to anyone.

---

### Agentic Architecture

Open Reality doesn't just map. It reasons about what it sees.

**Intent → Plan:** Before a frame is captured, the agent reads your goal and generates a typed spatial plan: which objects to find, what route to take, calibrated to your specific context. A firefighter's plan surfaces extinguishers and standpipe connections. A crime scene investigator's plan surfaces evidence markers. The agent understands the difference.

**Continuous Detection:** As the map grows, the agent automatically scans every new submap for your targets. No user action needed. The 3D world populates with labeled, located objects in real time.

**Retroactive Re-search:** Add a new target mid-scan — "find the AED too" — and the agent immediately re-runs detection on everything it's already seen. Its knowledge of the space updates instantly, backwards in time.

**Spatial Q&A:** After the scan, an AI assistant holds the full 3D context: every detected object, its exact position, the camera's path through the space. "Is the fire extinguisher accessible from the north stairwell?" gets answered with actual geometry — not a guess.

---

### Impact

Open Reality is a blueprint: what it looks like when you give AI agents a real sense of space and make it accessible to anyone. First responders, accessibility tools, robotics, construction, disaster response — any domain where knowing *where* something is in 3D space is a matter of safety.

The hardware barrier is gone. The deployment barrier is gone. What remains is the mission.

---

## Built with

Python · Modal · PyTorch · VGGT-1B · GTSAM · CLIP · SAM3 · DINO-Salad · Flask · Socket.IO · Three.js · Vite · TypeScript · Claude · Gemini

---

## "Try it out" links

GitHub: https://github.com/bdavidzhang/Real-Eyes

---

## Video demo concept — 2.5 minutes

**The thesis: make the viewer feel the stakes before showing the solution.**

---

**[0:00 – 0:20] — The Problem, Felt**

No voiceover. Just sound.

First-person handheld footage: someone walking through a cluttered space — an unfamiliar building, a dim hallway, a room with objects everywhere. Disoriented. Panning around. No mental map. Cut to black.

Text fades in: *"Where is the fire extinguisher?"*
Then: *"The AI doesn't know. It guesses."*

---

**[0:20 – 0:40] — Open Reality**

Clean browser UI. A prompt box. The product name.

Voiceover: *"Open Reality gives AI agents a real sense of space. Point any phone at a room. An agent builds a live 3D map — and finds exactly what you need."*

Someone types: *"I'm a paramedic doing a safety sweep of an office building."*

---

**[0:40 – 1:05] — The Plan**

The agent generates a spatial tracking plan. Object chips appear: fire extinguisher, AED, exit sign, stairwell access. A pathfinding rationale fills in below.

Voiceover: *"Before a frame is captured, the agent interprets your goal and builds a spatial action plan. What to find. How to move. Calibrated to your specific mission."*

User adds: *"wheelchair ramp."* It appears instantly. They hit Confirm.

---

**[1:05 – 1:30] — Phone → Modal → 3D Map**

Split screen. Left: phone browser, camera streamer. Right: the 3D viewer, empty, waiting.

The user starts. Status dot flips green. Frames stream. On the right, a point cloud materializes — first sparse, then dense as they walk. Camera frustums trace the path. A loop closure fires — the map snaps into alignment.

Voiceover: *"Every frame streams over WebSocket to an H100 on Modal. A 1B-parameter vision model predicts depth and camera pose in real time — no GPS, no depth sensor. Modal keeps inference hot. No cold starts. No dropped frames."*

Overlay: *"Modal · H100 GPU · one-command deploy"*

---

**[1:30 – 2:00] — The Agent Finds Things**

The detection panel populates. A red 3D bounding box appears around a fire extinguisher. A green one around an exit sign. The user clicks a result — a preview opens: the exact keyframe, the SAM3 segmentation mask.

Mid-scan, they type: *"AED defibrillator."*

The agent immediately re-runs on every previous submap. A new box appears — retroactively found.

Voiceover: *"The agent scores every submap against your targets using CLIP. When a match hits, SAM3 segments it in 2D and projects it into a precise 3D bounding box. Change your targets mid-scan — the agent re-searches everything it's already seen."*

---

**[2:00 – 2:20] — Spatial Q&A**

The scan ends. The Summary page opens. 2D floorplan on the left. Full 3D map with bounding boxes on the right.

The user types: *"Is the fire extinguisher accessible from the main entrance?"*

The AI responds using real spatial coordinates — not a guess.

Voiceover: *"When you're done, Open Reality gives you a full debrief. A floorplan. A 3D model. An AI that answers spatial questions using actual geometry."*

---

**[2:20 – 2:30] — Close**

Slow rotation of the completed 3D point cloud. Bounding boxes glow softly.

*"Open Reality."*
*"One phone. Any space. Real spatial intelligence."*
*"Deployed on Modal in one command."*

---

## Live demo plan (for judges)

1. **[0:20]** Cold open video — first-responder scenario. Establish the stakes.
2. **[0:30]** Live: type a mission prompt, watch the plan generate. Ask a judge to call out an object — add it live.
3. **[0:40]** Pre-recorded scan plays left screen, 3D map builds right. QR code on screen — judges open the live map on their phones.
4. **[0:30]** Detection results appear. Ask the AI a spatial question live.
5. **[0:10]** Close: "One command on Modal. Any phone. Open source."

**Total: ~2:30**

---

*HackIllinois 2026 — Modal Track*
