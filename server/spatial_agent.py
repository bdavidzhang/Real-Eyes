"""
SpatialAgent — Autonomous spatial intelligence agent for VGGT-SLAM 2.0.

Two-tier Gemini architecture:
  - Main orchestrator (Gemini 2.5 Pro): scene reasoning, mission management, narrative
  - Sub-agent tasks (Gemini 2.0 Flash): focused one-shot analysis (scene, objects, verification)

Mirrors the Claude Code agent pattern: orchestrator spawns focused sub-agents,
integrates results, maintains persistent mission state.
"""

import base64
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
except ImportError:
    genai = None


@dataclass
class Mission:
    id: int
    category: str
    goal: str
    queries: list
    found: set = field(default_factory=set)
    status: str = "active"  # active | completed | stalled
    findings: list = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    submaps_since_finding: int = 0


class SpatialAgent:
    def __init__(self, streaming_slam, emit_fn: Callable, gemini_api_key: str):
        self.slam = streaming_slam
        self.emit = emit_fn
        self.api_key = gemini_api_key

        # Models (lazy-initialized)
        self._pro_model = None
        self._flash_model = None
        self._configured = False

        # Scene understanding
        self.scene_description = ""
        self.room_type = "unknown"
        self.scene_history: list[dict] = []
        self.spatial_layout = ""

        # Mission tracking
        self.missions: dict[int, Mission] = {}
        self.next_mission_id = 1
        self.all_active_queries: list[str] = []

        # Detection tracking
        self.discovered_objects: dict[str, list] = {}
        self.previous_detections: list[dict] = []

        # User interaction
        self.current_goal: Optional[str] = None
        self.chat_history: list[dict] = []

        # Config
        self.enabled = True
        self.max_keyframes_per_analysis = 4
        self._processing = False

        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=3)

        # Submap counter for periodic tasks
        self._submap_count = 0

    def _ensure_configured(self):
        if not self._configured and genai is not None:
            genai.configure(api_key=self.api_key)
            self._configured = True

    @property
    def pro_model(self):
        if self._pro_model is None:
            self._ensure_configured()
            self._pro_model = genai.GenerativeModel('gemini-2.5-pro')
        return self._pro_model

    @property
    def flash_model(self):
        if self._flash_model is None:
            self._ensure_configured()
            self._flash_model = genai.GenerativeModel('gemini-3-flash-preview')
        return self._flash_model

    # ------------------------------------------------------------------
    # Entry point — called after each submap is processed
    # ------------------------------------------------------------------

    def on_submap_processed(self, submap_id: int):
        if not self.enabled or self._processing:
            return
        self._processing = True
        try:
            self._submap_count += 1
            self._emit_thought("Analyzing new area...", "thinking")

            # Extract keyframes from the latest submap
            keyframes_b64 = self._extract_keyframes_b64(submap_id)
            if not keyframes_b64:
                self._processing = False
                return

            # Get current detections
            with self.slam._detection_lock:
                current_detections = list(self.slam.accumulated_detections)

            # Run the agent cycle
            self._run_agent_cycle(keyframes_b64, current_detections, submap_id)

        except Exception as e:
            print(f"SpatialAgent error: {e}")
            import traceback
            traceback.print_exc()
            self._emit_thought(f"Analysis error, continuing...", "error")
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    # Core agent cycle
    # ------------------------------------------------------------------

    def _run_agent_cycle(self, keyframes_b64: list[str], detections: list[dict], submap_id: int):
        # Phase 1: Run scene analyzer + object spotter in parallel (Flash)
        scene_future = self._executor.submit(
            self._run_flash_task, "scene_analyzer", keyframes_b64
        )
        spotter_future = self._executor.submit(
            self._run_flash_task, "object_spotter", keyframes_b64
        )

        scene_result = self._safe_future_result(scene_future, "scene_analyzer")
        spotter_result = self._safe_future_result(spotter_future, "object_spotter")

        # Phase 2: Route new detections to missions
        new_detections = self._diff_detections(detections)
        if new_detections:
            self._route_detections(new_detections)

        # Phase 3: If spotter found objects and we have few/no missions, run category grouper
        spotted_objects = spotter_result.get("objects", []) if spotter_result else []
        existing_query_set = set(self.all_active_queries)
        new_objects = [
            obj["name"] for obj in spotted_objects
            if obj.get("name", "").lower() not in existing_query_set
        ]

        grouper_result = None
        if new_objects:
            grouper_result = self._run_flash_task(
                "category_grouper",
                keyframes_b64,
                extra_context={"objects": new_objects}
            )

        # Phase 4: Run orchestrator (Pro) to synthesize everything
        orchestrator_result = self._run_orchestrator(
            keyframes_b64, scene_result, spotter_result, grouper_result,
            new_detections, submap_id
        )

        if orchestrator_result:
            self._process_orchestrator_result(orchestrator_result)

        # Phase 5: Periodic layout mapping
        if self._submap_count % 5 == 0 and self._submap_count > 0:
            layout_future = self._executor.submit(
                self._run_flash_task, "layout_mapper", keyframes_b64,
                extra_context={
                    "camera_count": self.slam.frame_count,
                    "detected_objects": list(self.discovered_objects.keys()),
                }
            )
            layout_result = self._safe_future_result(layout_future, "layout_mapper")
            if layout_result:
                self.spatial_layout = layout_result.get("layout_description", "")
                if self.spatial_layout:
                    self._emit_thought(f"Spatial layout: {self.spatial_layout}", "observation")

        # Emit full state update
        self._emit_state()

    # ------------------------------------------------------------------
    # Orchestrator (Gemini 2.5 Pro)
    # ------------------------------------------------------------------

    def _run_orchestrator(self, keyframes_b64, scene_result, spotter_result,
                          grouper_result, new_detections, submap_id):
        missions_summary = []
        for m in self.missions.values():
            missions_summary.append({
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": list(m.found),
                "status": m.status,
                "confidence": m.confidence,
            })

        prompt = f"""You are the Spatial Intelligence Agent for a real-time 3D SLAM system.
You autonomously explore and catalog the physical environment.

CURRENT STATE:
- Submaps processed: {self._submap_count}
- Room type: {self.room_type}
- Scene so far: {self.scene_description}
- Active missions: {json.dumps(missions_summary)}
- Objects found so far: {json.dumps(list(self.discovered_objects.keys()))}
- User's goal: {self.current_goal or "autonomous exploration"}
- Recent chat: {json.dumps(self.chat_history[-3:]) if self.chat_history else "none"}

SUB-AGENT REPORTS THIS CYCLE:
- Scene Analyzer: {json.dumps(scene_result) if scene_result else "unavailable"}
- Object Spotter: {json.dumps(spotter_result) if spotter_result else "unavailable"}
- Category Grouper: {json.dumps(grouper_result) if grouper_result else "not run"}

NEW DETECTIONS (CLIP+SAM3 confirmed):
{json.dumps(new_detections[:10]) if new_detections else "none"}

Respond with ONLY valid JSON (no markdown):
{{
  "narrative": "1-3 sentences about what you see and what you're doing. Be specific and conversational.",
  "scene_update": "updated overall scene description",
  "room_type": "best guess room type",
  "new_missions": [
    {{"category": "Category Name", "goal": "What to find", "queries": ["query1", "query2"]}}
  ],
  "complete_missions": [],
  "add_queries_to_mission": {{}},
  "coverage_estimate": 0.0
}}"""

        try:
            # Build content with images
            content_parts = []
            for i, img_b64 in enumerate(keyframes_b64[:2]):
                img_data = base64.b64decode(img_b64)
                content_parts.append({
                    "mime_type": "image/jpeg",
                    "data": img_b64,
                })
            content_parts.append(prompt)

            response = self.pro_model.generate_content(
                content_parts,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                ),
            )
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Orchestrator error: {e}")
            # Fallback: use Flash as orchestrator
            try:
                response = self.flash_model.generate_content(
                    content_parts,
                    generation_config=GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    ),
                )
                return self._parse_json_response(response.text)
            except Exception as e2:
                print(f"Orchestrator fallback error: {e2}")
                return None

    def _process_orchestrator_result(self, result: dict):
        # Update scene description
        if result.get("scene_update"):
            self.scene_description = result["scene_update"]

        if result.get("room_type"):
            self.room_type = result["room_type"]

        # Emit narrative
        narrative = result.get("narrative", "")
        if narrative:
            self._emit_thought(narrative, "observation")

        # Create new missions
        for mission_data in result.get("new_missions", []):
            queries = mission_data.get("queries", [])
            if not queries:
                continue
            mission = Mission(
                id=self.next_mission_id,
                category=mission_data.get("category", "General"),
                goal=mission_data.get("goal", "Explore"),
                queries=[q.lower().strip() for q in queries],
            )
            self.missions[mission.id] = mission
            self.next_mission_id += 1
            self._emit_action(
                "mission_created",
                f"New mission: {mission.category} - {mission.goal}",
                {"mission_id": mission.id, "queries": mission.queries},
            )

        # Complete missions
        for mid in result.get("complete_missions", []):
            if mid in self.missions:
                self.missions[mid].status = "completed"
                m = self.missions[mid]
                self._emit_action(
                    "mission_completed",
                    f"Completed: {m.category} ({len(m.found)}/{len(m.queries)} found)",
                    {"mission_id": mid},
                )

        # Add queries to existing missions
        for mid_str, new_queries in result.get("add_queries_to_mission", {}).items():
            mid = int(mid_str)
            if mid in self.missions:
                for q in new_queries:
                    q_lower = q.lower().strip()
                    if q_lower not in self.missions[mid].queries:
                        self.missions[mid].queries.append(q_lower)

        # Update active queries across all missions
        self._sync_detection_queries()

        # Store scene history
        self.scene_history.append({
            "submap_count": self._submap_count,
            "description": self.scene_description,
            "timestamp": time.time(),
        })
        # Keep last 20
        self.scene_history = self.scene_history[-20:]

    # ------------------------------------------------------------------
    # Sub-agent tasks (Gemini 2.0 Flash)
    # ------------------------------------------------------------------

    def _run_flash_task(self, task_type: str, keyframes_b64: list[str],
                        extra_context: dict = None) -> Optional[dict]:
        prompts = {
            "scene_analyzer": (
                "Analyze these camera frames from a 3D scanning session. "
                "Return JSON: {\"description\": \"what you see in 1-2 sentences\", "
                "\"room_type\": \"office/kitchen/bedroom/bathroom/hallway/outdoor/other\", "
                "\"notable_features\": [\"feature1\", \"feature2\"]}"
            ),
            "object_spotter": (
                "List all distinct physical objects visible in these camera frames. "
                "Be specific and practical (e.g. 'office chair' not just 'furniture'). "
                "Return JSON: {\"objects\": [{\"name\": \"object_name\", "
                "\"count_estimate\": 1, \"location_hint\": \"where in frame\"}]}"
            ),
            "category_grouper": (
                "Group these detected objects into logical search categories/missions. "
                f"Objects: {json.dumps(extra_context.get('objects', [])) if extra_context else '[]'}. "
                "Return JSON: {\"groups\": [{\"category\": \"Category\", "
                "\"queries\": [\"query1\", \"query2\"], "
                "\"goal_description\": \"what to find\"}]}"
            ),
            "detail_verifier": (
                "Look at this frame. Is the highlighted region correctly identified? "
                f"Expected: {extra_context.get('query', '') if extra_context else ''}. "
                "Return JSON: {\"is_correct\": true/false, \"actual_object\": \"what it is\", "
                "\"details\": \"explanation\"}"
            ),
            "layout_mapper": (
                "Based on these frames and context, describe the spatial layout. "
                f"Context: {json.dumps(extra_context) if extra_context else '{}'}. "
                "Return JSON: {\"layout_description\": \"description of space layout\", "
                "\"room_dimensions_estimate\": \"rough size\", "
                "\"spatial_relationships\": [\"desk is near window\"]}"
            ),
        }

        prompt = prompts.get(task_type, "Describe what you see. Return JSON.")

        try:
            content_parts = []
            # For category_grouper, no images needed
            if task_type != "category_grouper":
                for img_b64 in keyframes_b64[:self.max_keyframes_per_analysis]:
                    content_parts.append({
                        "mime_type": "image/jpeg",
                        "data": img_b64,
                    })
            content_parts.append(prompt)

            response = self.flash_model.generate_content(
                content_parts,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Flash task '{task_type}' error: {e}")
            return None

    # ------------------------------------------------------------------
    # Detection routing
    # ------------------------------------------------------------------

    def _diff_detections(self, current_detections: list[dict]) -> list[dict]:
        prev_keys = set()
        for d in self.previous_detections:
            key = (d.get("query", ""), d.get("matched_submap"), d.get("matched_frame"))
            prev_keys.add(key)

        new_detections = []
        for d in current_detections:
            key = (d.get("query", ""), d.get("matched_submap"), d.get("matched_frame"))
            if key not in prev_keys:
                new_detections.append(d)

        self.previous_detections = current_detections
        return new_detections

    def _route_detections(self, new_detections: list[dict]):
        for det in new_detections:
            query = det.get("query", "").lower()
            confidence = det.get("confidence", 0)

            # Track in discovered objects
            if query not in self.discovered_objects:
                self.discovered_objects[query] = []
            self.discovered_objects[query].append(det)

            # Route to matching mission
            for mission in self.missions.values():
                if mission.status != "active":
                    continue
                if query in mission.queries:
                    mission.found.add(query)
                    mission.findings.append(det)
                    mission.submaps_since_finding = 0
                    mission.confidence = len(mission.found) / max(len(mission.queries), 1)

            # Emit finding
            self._emit_finding(
                query=query,
                description=f"Found: {query} (confidence: {confidence:.0%})",
                confidence=confidence,
                position=det.get("bounding_box", {}).get("center") if det.get("bounding_box") else None,
                mission_id=self._find_mission_for_query(query),
            )

        # Update mission stale counters
        for mission in self.missions.values():
            if mission.status == "active":
                has_new = any(
                    d.get("query", "").lower() in mission.queries
                    for d in new_detections
                )
                if not has_new:
                    mission.submaps_since_finding += 1

                # Auto-complete logic
                if (mission.submaps_since_finding >= 5
                        and len(mission.found) > 0
                        and mission.confidence >= 0.5):
                    mission.status = "completed"
                    self._emit_action(
                        "mission_completed",
                        f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                        {"mission_id": mission.id},
                    )

    def _find_mission_for_query(self, query: str) -> Optional[int]:
        for mission in self.missions.values():
            if query in mission.queries:
                return mission.id
        return None

    # ------------------------------------------------------------------
    # Detection query sync
    # ------------------------------------------------------------------

    def _sync_detection_queries(self):
        all_queries = set()
        for mission in self.missions.values():
            if mission.status == "active":
                for q in mission.queries:
                    all_queries.add(q)

        new_queries = sorted(all_queries)
        if new_queries != self.all_active_queries:
            self.all_active_queries = new_queries
            # Update SLAM detection queries
            try:
                self.slam.set_detection_queries(new_queries)
                if new_queries:
                    self._emit_action(
                        "queries_updated",
                        f"Searching for: {', '.join(new_queries)}",
                        {"queries": new_queries},
                    )
            except Exception as e:
                print(f"Error setting detection queries: {e}")

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------

    def handle_user_message(self, message: str) -> str:
        self.chat_history.append({"role": "user", "content": message})

        missions_summary = []
        for m in self.missions.values():
            missions_summary.append({
                "id": m.id, "category": m.category, "goal": m.goal,
                "queries": m.queries, "found": list(m.found), "status": m.status,
            })

        prompt = f"""You are the Spatial Intelligence Agent. A user is chatting with you during a live 3D scan.

SCENE CONTEXT:
- Room type: {self.room_type}
- Scene: {self.scene_description}
- Objects found: {json.dumps(list(self.discovered_objects.keys()))}
- Active missions: {json.dumps(missions_summary)}
- Submaps processed: {self._submap_count}

CHAT HISTORY:
{json.dumps(self.chat_history[-5:])}

USER MESSAGE: {message}

Respond with JSON:
{{
  "response": "Your conversational reply to the user",
  "new_missions": [
    {{"category": "Category", "goal": "Goal", "queries": ["q1", "q2"]}}
  ],
  "remove_queries": [],
  "set_goal": null
}}

If the user asks to find something, create a new mission.
If the user asks to stop searching for something, add to remove_queries.
If the user sets a goal, put it in set_goal."""

        try:
            response = self.pro_model.generate_content(
                prompt,
                generation_config=GenerationConfig(temperature=0.7, max_output_tokens=512),
            )
            result = self._parse_json_response(response.text)

            reply = result.get("response", "I'm still analyzing the scene.") if result else "I'm still analyzing the scene."

            # Process new missions
            if result:
                for mission_data in result.get("new_missions", []):
                    queries = mission_data.get("queries", [])
                    if queries:
                        mission = Mission(
                            id=self.next_mission_id,
                            category=mission_data.get("category", "User Request"),
                            goal=mission_data.get("goal", message),
                            queries=[q.lower().strip() for q in queries],
                        )
                        self.missions[mission.id] = mission
                        self.next_mission_id += 1
                        self._emit_action(
                            "mission_created",
                            f"New mission: {mission.category}",
                            {"mission_id": mission.id, "queries": mission.queries},
                        )

                # Handle remove queries
                for q in result.get("remove_queries", []):
                    q_lower = q.lower().strip()
                    for mission in self.missions.values():
                        if q_lower in mission.queries:
                            mission.queries.remove(q_lower)

                # Handle goal setting
                if result.get("set_goal"):
                    self.current_goal = result["set_goal"]

                self._sync_detection_queries()

            self.chat_history.append({"role": "assistant", "content": reply})
            self._emit_thought(reply, "chat_response")
            self._emit_state()
            return reply

        except Exception as e:
            print(f"Chat error: {e}")
            reply = "I'm having trouble processing that right now. Let me continue scanning."
            self.chat_history.append({"role": "assistant", "content": reply})
            self._emit_thought(reply, "chat_response")
            return reply

    def set_goal(self, goal: str):
        self.current_goal = goal
        self._emit_thought(f"New goal set: {goal}", "action")

        # Trigger a goal planner sub-agent
        result = self._run_flash_task(
            "object_spotter", [],
            extra_context={"goal": goal}
        )
        # The next cycle will pick up the goal context

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit_thought(self, content: str, thought_type: str = "observation"):
        data = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "type": thought_type,
            "content": content,
        }
        try:
            self.emit("agent_thought", data)
        except Exception as e:
            print(f"Emit error (thought): {e}")

    def _emit_action(self, action: str, details: str, extra: dict = None):
        data = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "action": action,
            "details": details,
        }
        if extra:
            data.update(extra)
        try:
            self.emit("agent_action", data)
        except Exception as e:
            print(f"Emit error (action): {e}")

    def _emit_finding(self, query: str, description: str, confidence: float,
                      position: Optional[list] = None, mission_id: Optional[int] = None):
        data = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "query": query,
            "description": description,
            "confidence": confidence,
            "position": position,
            "mission_id": mission_id,
        }
        try:
            self.emit("agent_finding", data)
        except Exception as e:
            print(f"Emit error (finding): {e}")

    def _emit_state(self):
        try:
            self.emit("agent_state", self.get_state())
        except Exception as e:
            print(f"Emit error (state): {e}")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        missions_list = []
        for m in self.missions.values():
            missions_list.append({
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": list(m.found),
                "status": m.status,
                "confidence": m.confidence,
                "findings_count": len(m.findings),
            })

        return {
            "enabled": self.enabled,
            "scene_description": self.scene_description,
            "room_type": self.room_type,
            "missions": missions_list,
            "active_queries": self.all_active_queries,
            "discovered_objects": list(self.discovered_objects.keys()),
            "current_goal": self.current_goal,
            "submaps_processed": self._submap_count,
            "coverage_estimate": min(self._submap_count * 0.1, 1.0),
        }

    def reset(self):
        with self._lock:
            self.scene_description = ""
            self.room_type = "unknown"
            self.scene_history.clear()
            self.spatial_layout = ""
            self.missions.clear()
            self.next_mission_id = 1
            self.all_active_queries.clear()
            self.discovered_objects.clear()
            self.previous_detections.clear()
            self.current_goal = None
            self.chat_history.clear()
            self._submap_count = 0
            self._processing = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_keyframes_b64(self, submap_id: int) -> list[str]:
        try:
            submap = self.slam.solver.map.get_submap(submap_id)
            if submap is None:
                # Try latest submap
                submap = self.slam.solver.map.get_latest_submap()
            if submap is None:
                return []

            keyframes = []
            num_frames = submap.get_num_frames()
            # Sample up to max_keyframes evenly
            indices = list(range(0, num_frames, max(1, num_frames // self.max_keyframes_per_analysis)))
            indices = indices[:self.max_keyframes_per_analysis]

            for idx in indices:
                try:
                    frame_tensor = submap.get_frame_at_index(idx)
                    frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # Resize to 512px wide for efficiency
                    h, w = frame_np.shape[:2]
                    if w > 512:
                        scale = 512 / w
                        frame_np = cv2.resize(frame_np, (512, int(h * scale)))
                    # Convert RGB to BGR for cv2 encoding
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    ok, jpeg_buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        keyframes.append(base64.b64encode(jpeg_buf.tobytes()).decode('ascii'))
                except Exception as e:
                    print(f"Keyframe extraction error (idx={idx}): {e}")

            return keyframes
        except Exception as e:
            print(f"Keyframe extraction error: {e}")
            return []

    def _parse_json_response(self, text: str) -> Optional[dict]:
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to extract JSON from markdown code blocks or surrounding text
        match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find any JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"Failed to parse JSON from: {text[:200]}")
        return None

    def _safe_future_result(self, future, task_name: str, timeout: float = 15.0):
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            print(f"Flash task '{task_name}' timed out")
            return None
        except Exception as e:
            print(f"Flash task '{task_name}' error: {e}")
            return None
