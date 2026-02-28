"""SpatialAgent: autonomous spatial intelligence with OpenRouter-backed subagents."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from server.llm import OpenRouterClient


@dataclass
class Mission:
    id: int
    category: str
    goal: str
    queries: list[str]
    found: set[str] = field(default_factory=set)
    status: str = "active"  # active | completed | stalled
    findings: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    submaps_since_finding: int = 0


class SpatialAgent:
    """Session-scoped autonomous agent for SLAM scene exploration."""

    def __init__(
        self,
        streaming_slam,
        emit_fn: Callable[[str, dict[str, Any]], None],
        openrouter_api_key: str,
        session_id: str = "global",
        on_queries_changed: Optional[Callable[[str, list[str]], None]] = None,
    ):
        self.slam = streaming_slam
        self.emit = emit_fn
        self.session_id = session_id
        self.on_queries_changed = on_queries_changed

        # LLM config
        orch_model = os.environ.get(
            "SPATIAL_ORCH_MODEL", "anthropic/claude-3.5-sonnet-20241022"
        )
        sub_model = os.environ.get(
            "SPATIAL_SUBAGENT_MODEL", "anthropic/claude-3.5-haiku-20241022"
        )
        orch_fallbacks = self._parse_csv_env(
            "SPATIAL_ORCH_FALLBACKS",
            "anthropic/claude-3.5-haiku-20241022,openai/gpt-4o-mini",
        )
        sub_fallbacks = self._parse_csv_env(
            "SPATIAL_SUBAGENT_FALLBACKS",
            "openai/gpt-4o-mini",
        )

        self.orchestrator_client = OpenRouterClient(
            api_key=openrouter_api_key,
            primary_model=orch_model,
            fallback_models=orch_fallbacks,
            timeout=float(os.environ.get("SPATIAL_LLM_TIMEOUT_S", "20")),
            app_name="Real-Eyes Spatial Agent",
            max_retries=int(os.environ.get("SPATIAL_LLM_RETRIES", "2")),
        )
        self.subagent_client = OpenRouterClient(
            api_key=openrouter_api_key,
            primary_model=sub_model,
            fallback_models=sub_fallbacks,
            timeout=float(os.environ.get("SPATIAL_LLM_TIMEOUT_S", "20")),
            app_name="Real-Eyes Spatial Subagents",
            max_retries=int(os.environ.get("SPATIAL_LLM_RETRIES", "2")),
        )

        # Guardrails
        self.max_missions = int(os.environ.get("SPATIAL_MAX_MISSIONS", "8"))
        self.max_active_queries = int(os.environ.get("SPATIAL_MAX_ACTIVE_QUERIES", "12"))
        self.max_keyframes_per_analysis = int(os.environ.get("SPATIAL_MAX_KEYFRAMES", "4"))
        self.query_update_cooldown_s = float(os.environ.get("SPATIAL_QUERY_COOLDOWN_S", "2.0"))
        self.cycle_min_interval_s = float(os.environ.get("SPATIAL_CYCLE_MIN_INTERVAL_S", "1.2"))
        self.auto_complete_submap_gap = int(os.environ.get("SPATIAL_AUTOCOMPLETE_GAP", "6"))

        # Scene understanding
        self.scene_description = ""
        self.room_type = "unknown"
        self.spatial_layout = ""
        self.scene_history: list[dict[str, Any]] = []

        # Mission tracking
        self.missions: dict[int, Mission] = {}
        self.next_mission_id = 1
        self.all_active_queries: list[str] = []

        # Detection tracking
        self.discovered_objects: dict[str, list[dict[str, Any]]] = {}
        self.previous_detection_keys: set[tuple[str, int, int]] = set()

        # User interaction
        self.current_goal: Optional[str] = None
        self.chat_history: list[dict[str, str]] = []

        # Visual context memory (bounded session cache)
        self.visual_memory: deque[dict[str, Any]] = deque(maxlen=24)

        # Runtime
        self.enabled = True
        self._processing = False
        self._last_query_sync_ts = 0.0
        self._last_cycle_ts = 0.0
        self._submap_count = 0
        self._coverage_estimate = 0.0

        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=3)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def on_submap_processed(self, submap_id: int, detections: Optional[list[dict[str, Any]]] = None):
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_cycle_ts < self.cycle_min_interval_s:
            return

        with self._lock:
            if self._processing:
                return
            self._processing = True
            self._last_cycle_ts = now

        try:
            self._submap_count = max(self._submap_count, submap_id + 1)
            keyframes_b64 = self._extract_keyframes_b64(submap_id)
            if keyframes_b64:
                self._remember_keyframes(submap_id, keyframes_b64)

            if detections is None:
                with self.slam._detection_lock:
                    detections = list(self.slam.accumulated_detections)

            new_detections = self._diff_detections(detections or [])
            if new_detections:
                self._route_detections(new_detections)

            # Subagents in parallel
            scene_future = self._executor.submit(self._run_scene_analyzer, keyframes_b64)
            spotter_future = self._executor.submit(self._run_object_spotter, keyframes_b64)

            scene_result = self._safe_future_result(scene_future, "scene_analyzer")
            spotter_result = self._safe_future_result(spotter_future, "object_spotter")

            layout_result = None
            if self._submap_count % 4 == 0:
                layout_result = self._run_layout_mapper(keyframes_b64)
                if layout_result and layout_result.get("layout_description"):
                    self.spatial_layout = layout_result["layout_description"]

            self._bootstrap_missions_from_spotter(spotter_result)

            orchestrator_result = self._run_orchestrator(
                keyframes_b64=keyframes_b64,
                scene_result=scene_result,
                spotter_result=spotter_result,
                layout_result=layout_result,
                new_detections=new_detections,
            )
            if orchestrator_result:
                self._apply_orchestrator_result(orchestrator_result)

            # Coverage estimate: mostly based on explored submaps and mission completion.
            completed = sum(1 for m in self.missions.values() if m.status == "completed")
            total = max(len(self.missions), 1)
            mission_ratio = completed / total
            self._coverage_estimate = min(1.0, (self._submap_count * 0.08) + (mission_ratio * 0.25))

            self._emit_state()

        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] cycle error: {e}")
            self._emit_thought(
                "Agent hit a transient issue and will continue scanning.",
                thought_type="error",
                subagent="orchestrator",
            )
        finally:
            with self._lock:
                self._processing = False

    # ------------------------------------------------------------------
    # Orchestrator + subagents
    # ------------------------------------------------------------------

    def _run_scene_analyzer(self, keyframes_b64: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are a scene analysis subagent for a live 3D SLAM system. "
            "Output strict JSON only."
        )
        user_prompt = (
            "Analyze the provided keyframes. Return JSON: "
            "{\"description\":\"1-2 concise sentences\","
            "\"room_type\":\"office|kitchen|bedroom|bathroom|hallway|outdoor|other\","
            "\"notable_features\":[\"...\"]}"
        )

        try:
            parsed, _ = self.subagent_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[: self.max_keyframes_per_analysis],
                temperature=0.2,
                max_tokens=384,
            )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] scene analyzer error: {e}")
            return None

    def _run_object_spotter(self, keyframes_b64: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are an object spotting subagent for a real-world mapping system. "
            "List only concrete visible objects. Output strict JSON only."
        )
        user_prompt = (
            "Return JSON: {\"objects\":[{\"name\":\"object name\","
            "\"count_estimate\":1,\"location_hint\":\"where\"}]}."
        )

        try:
            parsed, _ = self.subagent_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[: self.max_keyframes_per_analysis],
                temperature=0.2,
                max_tokens=448,
            )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] object spotter error: {e}")
            return None

    def _run_layout_mapper(self, keyframes_b64: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "You are a spatial layout subagent. Infer rough layout relationships from keyframes. "
            "Output strict JSON only."
        )
        user_prompt = (
            "Return JSON: {\"layout_description\":\"...\","
            "\"spatial_relationships\":[\"item A is near item B\"],"
            "\"room_dimensions_estimate\":\"rough dimensions\"}."
        )

        try:
            parsed, _ = self.subagent_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[: self.max_keyframes_per_analysis],
                temperature=0.25,
                max_tokens=512,
            )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] layout mapper error: {e}")
            return None

    def _run_orchestrator(
        self,
        keyframes_b64: list[str],
        scene_result: Optional[dict[str, Any]],
        spotter_result: Optional[dict[str, Any]],
        layout_result: Optional[dict[str, Any]],
        new_detections: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        missions_summary = [
            {
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": sorted(m.found),
                "status": m.status,
                "confidence": m.confidence,
            }
            for m in self.missions.values()
        ]

        system_prompt = (
            "You are the orchestrator for an autonomous spatial intelligence system. "
            "You must reason conservatively, avoid hallucinations, and only create practical"
            " object-search missions grounded in visible scene context. Output strict JSON only."
        )

        user_prompt = (
            "Context:\n"
            f"- Session: {self.session_id}\n"
            f"- Submaps processed: {self._submap_count}\n"
            f"- Room type: {self.room_type}\n"
            f"- Scene description: {self.scene_description}\n"
            f"- Spatial layout: {self.spatial_layout}\n"
            f"- User goal: {self.current_goal or 'autonomous exploration'}\n"
            f"- Active missions: {json.dumps(missions_summary)}\n"
            f"- Objects discovered: {json.dumps(list(self.discovered_objects.keys()))}\n"
            f"- New detections this cycle: {json.dumps(new_detections[:12])}\n"
            f"- Scene analyzer: {json.dumps(scene_result) if scene_result else 'unavailable'}\n"
            f"- Object spotter: {json.dumps(spotter_result) if spotter_result else 'unavailable'}\n"
            f"- Layout mapper: {json.dumps(layout_result) if layout_result else 'not run'}\n"
            "\nReturn JSON exactly in this shape:\n"
            "{"
            "\"narrative\":\"1-3 concise sentences\"," 
            "\"scene_update\":\"updated scene summary\","
            "\"room_type\":\"office|kitchen|bedroom|bathroom|hallway|outdoor|other\","
            "\"new_missions\":[{\"category\":\"Category\",\"goal\":\"Goal\",\"queries\":[\"q1\",\"q2\"]}],"
            "\"complete_missions\":[1],"
            "\"add_queries_to_mission\":{\"2\":[\"q3\"]},"
            "\"coverage_estimate\":0.0"
            "}"
        )

        try:
            parsed, response = self.orchestrator_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=keyframes_b64[:2],
                temperature=0.35,
                max_tokens=1024,
            )
            if response.degraded:
                self._emit_thought(
                    f"LLM fallback active ({response.model}) — continuing in degraded mode.",
                    thought_type="thinking",
                    subagent="orchestrator",
                )
            return parsed
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] orchestrator error: {e}")
            return None

    def _bootstrap_missions_from_spotter(self, spotter_result: Optional[dict[str, Any]]):
        if not spotter_result or len(self.missions) >= 2:
            return

        objects = spotter_result.get("objects", [])
        candidate_queries: list[str] = []
        for obj in objects:
            name = str(obj.get("name", "")).strip().lower()
            if not name:
                continue
            if name not in candidate_queries:
                candidate_queries.append(name)
            if len(candidate_queries) >= 4:
                break

        if not candidate_queries:
            return

        self._create_mission(
            category="Autonomous Recon",
            goal="Catalog key objects in the current environment",
            queries=candidate_queries,
        )

    def _apply_orchestrator_result(self, result: dict[str, Any]):
        scene_update = result.get("scene_update")
        if isinstance(scene_update, str) and scene_update.strip():
            self.scene_description = scene_update.strip()

        room_type = result.get("room_type")
        if isinstance(room_type, str) and room_type.strip():
            self.room_type = room_type.strip().lower()

        narrative = result.get("narrative")
        if isinstance(narrative, str) and narrative.strip():
            self._emit_thought(
                narrative.strip(),
                thought_type="observation",
                subagent="orchestrator",
                keyframe_b64=self._latest_keyframe_b64(),
            )

        new_missions = result.get("new_missions", [])
        if isinstance(new_missions, list):
            for mission_data in new_missions:
                if not isinstance(mission_data, dict):
                    continue
                queries = mission_data.get("queries", [])
                if not isinstance(queries, list):
                    continue
                self._create_mission(
                    category=str(mission_data.get("category", "General")).strip() or "General",
                    goal=str(mission_data.get("goal", "Explore")).strip() or "Explore",
                    queries=[str(q) for q in queries],
                )

        complete = result.get("complete_missions", [])
        if isinstance(complete, list):
            for mid in complete:
                try:
                    mission_id = int(mid)
                except Exception:
                    continue
                mission = self.missions.get(mission_id)
                if mission is None:
                    continue
                mission.status = "completed"
                self._emit_action(
                    "mission_completed",
                    f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                    {"mission_id": mission_id},
                )

        add_queries = result.get("add_queries_to_mission", {})
        if isinstance(add_queries, dict):
            for mid_str, query_list in add_queries.items():
                if not isinstance(query_list, list):
                    continue
                try:
                    mission_id = int(mid_str)
                except Exception:
                    continue
                mission = self.missions.get(mission_id)
                if mission is None or mission.status != "active":
                    continue
                for query in query_list:
                    q = str(query).strip().lower()
                    if q and q not in mission.queries:
                        mission.queries.append(q)

        cov = result.get("coverage_estimate")
        if isinstance(cov, (int, float)):
            self._coverage_estimate = min(max(float(cov), 0.0), 1.0)

        self._sync_detection_queries()

        self.scene_history.append(
            {
                "submap_count": self._submap_count,
                "scene_description": self.scene_description,
                "room_type": self.room_type,
                "timestamp": time.time(),
            }
        )
        self.scene_history = self.scene_history[-20:]

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------

    def handle_user_message(self, message: str) -> str:
        message = (message or "").strip()
        if not message:
            return ""

        self.chat_history.append({"role": "user", "content": message})
        self.chat_history = self.chat_history[-20:]

        missions_summary = [
            {
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": sorted(m.found),
                "status": m.status,
            }
            for m in self.missions.values()
        ]

        system_prompt = (
            "You are the session-specific Spatial Intelligence Agent for a live 3D scan. "
            "Be concise, practical, and mission-driven. Output strict JSON only."
        )
        user_prompt = (
            "State:\n"
            f"- Room type: {self.room_type}\n"
            f"- Scene: {self.scene_description}\n"
            f"- Layout: {self.spatial_layout}\n"
            f"- Objects: {json.dumps(list(self.discovered_objects.keys()))}\n"
            f"- Missions: {json.dumps(missions_summary)}\n"
            f"- Goal: {self.current_goal or 'none'}\n"
            f"- Recent chat: {json.dumps(self.chat_history[-6:])}\n"
            f"\nUser message: {message}\n\n"
            "Return JSON:\n"
            "{"
            "\"response\":\"assistant reply\","
            "\"new_missions\":[{\"category\":\"...\",\"goal\":\"...\",\"queries\":[\"q1\"]}],"
            "\"remove_queries\":[\"q\"],"
            "\"set_goal\":null"
            "}"
        )

        try:
            parsed, _ = self.orchestrator_client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=self._recent_context_images(max_images=2),
                temperature=0.45,
                max_tokens=768,
            )

            reply = str(parsed.get("response", "I am continuing to scan and organize the scene.")).strip()

            for mission_data in parsed.get("new_missions", []):
                if not isinstance(mission_data, dict):
                    continue
                self._create_mission(
                    category=str(mission_data.get("category", "User Request")),
                    goal=str(mission_data.get("goal", message)),
                    queries=[str(q) for q in mission_data.get("queries", [])],
                )

            for q in parsed.get("remove_queries", []):
                q_lower = str(q).strip().lower()
                if not q_lower:
                    continue
                for mission in self.missions.values():
                    if q_lower in mission.queries:
                        mission.queries.remove(q_lower)

            set_goal = parsed.get("set_goal")
            if isinstance(set_goal, str) and set_goal.strip():
                self.current_goal = set_goal.strip()

            self._sync_detection_queries()
            self.chat_history.append({"role": "assistant", "content": reply})
            self.chat_history = self.chat_history[-20:]

            self._emit_thought(
                reply,
                thought_type="chat_response",
                subagent="orchestrator",
                keyframe_b64=self._latest_keyframe_b64(),
            )
            self._emit_state()
            return reply

        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] chat error: {e}")
            fallback = "I hit a temporary model issue, but I am still scanning and updating missions."
            self.chat_history.append({"role": "assistant", "content": fallback})
            self.chat_history = self.chat_history[-20:]
            self._emit_thought(fallback, thought_type="error", subagent="orchestrator")
            return fallback

    def set_goal(self, goal: str):
        self.current_goal = (goal or "").strip() or None
        if self.current_goal:
            self._emit_action("goal_updated", f"New goal set: {self.current_goal}")
        self._emit_state()

    # ------------------------------------------------------------------
    # Detection routing and query sync
    # ------------------------------------------------------------------

    def _diff_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new: list[dict[str, Any]] = []
        current_keys: set[tuple[str, int, int]] = set()

        for det in detections:
            query = str(det.get("query", "")).lower()
            submap = int(det.get("matched_submap", -1))
            frame = int(det.get("matched_frame", -1))
            key = (query, submap, frame)
            current_keys.add(key)
            if key not in self.previous_detection_keys:
                new.append(det)

        self.previous_detection_keys = current_keys
        return new

    def _route_detections(self, new_detections: list[dict[str, Any]]):
        for det in new_detections:
            query = str(det.get("query", "")).strip().lower()
            if not query:
                continue

            confidence = float(det.get("confidence", 0.0) or 0.0)
            self.discovered_objects.setdefault(query, []).append(det)

            for mission in self.missions.values():
                if mission.status != "active":
                    continue
                if query in mission.queries:
                    mission.found.add(query)
                    mission.findings.append(det)
                    mission.submaps_since_finding = 0
                    mission.confidence = len(mission.found) / max(1, len(mission.queries))

            evidence = {
                "matched_submap": det.get("matched_submap"),
                "matched_frame": det.get("matched_frame"),
            }
            self._emit_finding(
                query=query,
                description=f"Found {query} ({confidence:.0%})",
                confidence=confidence,
                position=det.get("bounding_box", {}).get("center") if det.get("bounding_box") else None,
                mission_id=self._find_mission_for_query(query),
                evidence=evidence,
            )

        for mission in self.missions.values():
            if mission.status != "active":
                continue

            has_new = any(str(d.get("query", "")).strip().lower() in mission.queries for d in new_detections)
            if not has_new:
                mission.submaps_since_finding += 1

            if (
                mission.submaps_since_finding >= self.auto_complete_submap_gap
                and len(mission.found) > 0
                and mission.confidence >= 0.5
            ):
                mission.status = "completed"
                self._emit_action(
                    "mission_completed",
                    f"Completed: {mission.category} ({len(mission.found)}/{len(mission.queries)})",
                    {"mission_id": mission.id},
                )

    def _sync_detection_queries(self):
        active_queries: list[str] = []
        for mission in self.missions.values():
            if mission.status != "active":
                continue
            for q in mission.queries:
                q_norm = q.strip().lower()
                if q_norm and q_norm not in active_queries:
                    active_queries.append(q_norm)

        active_queries = active_queries[: self.max_active_queries]
        if active_queries == self.all_active_queries:
            return

        now = time.time()
        if now - self._last_query_sync_ts < self.query_update_cooldown_s:
            return

        self._last_query_sync_ts = now
        self.all_active_queries = active_queries

        if self.on_queries_changed is not None:
            try:
                self.on_queries_changed(self.session_id, list(self.all_active_queries))
            except Exception as e:
                print(f"SpatialAgent[{self.session_id}] query callback error: {e}")

        details = ", ".join(self.all_active_queries) if self.all_active_queries else "none"
        self._emit_action(
            "queries_updated",
            f"Searching for: {details}",
            {"queries": list(self.all_active_queries)},
        )

    # ------------------------------------------------------------------
    # State + reset
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        missions_list = [
            {
                "id": m.id,
                "category": m.category,
                "goal": m.goal,
                "queries": m.queries,
                "found": sorted(m.found),
                "status": m.status,
                "confidence": m.confidence,
                "findings_count": len(m.findings),
            }
            for m in self.missions.values()
        ]

        return {
            "enabled": self.enabled,
            "scene_description": self.scene_description,
            "room_type": self.room_type,
            "missions": missions_list,
            "active_queries": list(self.all_active_queries),
            "discovered_objects": sorted(self.discovered_objects.keys()),
            "current_goal": self.current_goal,
            "submaps_processed": self._submap_count,
            "coverage_estimate": min(max(self._coverage_estimate, 0.0), 1.0),
            "health": "degraded" if self.orchestrator_client.degraded_mode else "ok",
            "degraded_mode": self.orchestrator_client.degraded_mode,
        }

    def reset(self):
        with self._lock:
            self.scene_description = ""
            self.room_type = "unknown"
            self.spatial_layout = ""
            self.scene_history.clear()
            self.missions.clear()
            self.next_mission_id = 1
            self.all_active_queries.clear()
            self.discovered_objects.clear()
            self.previous_detection_keys.clear()
            self.current_goal = None
            self.chat_history.clear()
            self.visual_memory.clear()
            self._processing = False
            self._submap_count = 0
            self._coverage_estimate = 0.0
            self._last_query_sync_ts = 0.0
            self._last_cycle_ts = 0.0

        if self.on_queries_changed is not None:
            try:
                self.on_queries_changed(self.session_id, [])
            except Exception as e:
                print(f"SpatialAgent[{self.session_id}] reset callback error: {e}")

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit_thought(
        self,
        content: str,
        thought_type: str = "observation",
        subagent: Optional[str] = None,
        confidence: Optional[float] = None,
        keyframe_b64: Optional[str] = None,
    ):
        data: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "type": thought_type,
            "content": content,
        }
        if subagent:
            data["subagent"] = subagent
        if confidence is not None:
            data["confidence"] = float(confidence)
        if keyframe_b64:
            data["keyframe_b64"] = keyframe_b64
            data["attachments"] = [{"kind": "keyframe", "image_b64": keyframe_b64}]
        try:
            self.emit("agent_thought", data)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit thought error: {e}")

    def _emit_action(self, action: str, details: str, extra: Optional[dict[str, Any]] = None):
        data: dict[str, Any] = {
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
            print(f"SpatialAgent[{self.session_id}] emit action error: {e}")

    def _emit_finding(
        self,
        query: str,
        description: str,
        confidence: float,
        position: Optional[list] = None,
        mission_id: Optional[int] = None,
        evidence: Optional[dict[str, Any]] = None,
    ):
        data: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "query": query,
            "description": description,
            "confidence": confidence,
            "position": position,
            "mission_id": mission_id,
        }
        if evidence:
            data["evidence"] = evidence
        try:
            self.emit("agent_finding", data)
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit finding error: {e}")

    def _emit_state(self):
        try:
            self.emit("agent_state", self.get_state())
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] emit state error: {e}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _create_mission(self, category: str, goal: str, queries: list[str]):
        normalized = []
        for query in queries:
            q = str(query).strip().lower()
            if q and q not in normalized:
                normalized.append(q)

        if not normalized:
            return

        if len(self.missions) >= self.max_missions:
            return

        mission = Mission(
            id=self.next_mission_id,
            category=category,
            goal=goal,
            queries=normalized,
        )
        self.missions[mission.id] = mission
        self.next_mission_id += 1

        self._emit_action(
            "mission_created",
            f"New mission: {mission.category} - {mission.goal}",
            {"mission_id": mission.id, "queries": mission.queries},
        )

    def _extract_keyframes_b64(self, submap_id: int) -> list[str]:
        try:
            submap = self.slam.solver.map.get_submap(submap_id)
            if submap is None:
                submap = self.slam.solver.map.get_latest_submap()
            if submap is None:
                return []

            num_frames = submap.get_num_frames()
            if num_frames <= 0:
                return []

            step = max(1, num_frames // self.max_keyframes_per_analysis)
            indices = list(range(0, num_frames, step))[: self.max_keyframes_per_analysis]
            keyframes: list[str] = []

            for idx in indices:
                try:
                    frame_tensor = submap.get_frame_at_index(idx)
                    frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    h, w = frame_np.shape[:2]
                    if w > 640:
                        scale = 640.0 / float(w)
                        frame_np = cv2.resize(frame_np, (640, int(h * scale)))

                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    ok, jpeg_buf = cv2.imencode(
                        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
                    )
                    if ok:
                        import base64

                        keyframes.append(base64.b64encode(jpeg_buf.tobytes()).decode("ascii"))
                except Exception as e:
                    print(f"SpatialAgent[{self.session_id}] keyframe extraction error idx={idx}: {e}")

            return keyframes
        except Exception as e:
            print(f"SpatialAgent[{self.session_id}] keyframe extraction error: {e}")
            return []

    def _remember_keyframes(self, submap_id: int, keyframes_b64: list[str]):
        ts = time.time()
        for b64 in keyframes_b64:
            self.visual_memory.append(
                {
                    "timestamp": ts,
                    "submap_id": submap_id,
                    "image_b64": b64,
                }
            )

    def _recent_context_images(self, max_images: int = 4) -> list[str]:
        images: list[str] = []
        seen = set()
        for item in reversed(self.visual_memory):
            img = item.get("image_b64")
            if not img or img in seen:
                continue
            images.append(img)
            seen.add(img)
            if len(images) >= max_images:
                break
        return images

    def _latest_keyframe_b64(self) -> Optional[str]:
        if not self.visual_memory:
            return None
        latest = self.visual_memory[-1]
        img = latest.get("image_b64")
        return img if isinstance(img, str) else None

    def _find_mission_for_query(self, query: str) -> Optional[int]:
        for mission in self.missions.values():
            if query in mission.queries:
                return mission.id
        return None

    @staticmethod
    def _parse_csv_env(env_name: str, default: str) -> list[str]:
        raw = os.environ.get(env_name, default)
        return [part.strip() for part in raw.split(",") if part.strip()]

    @staticmethod
    def _safe_future_result(future, task_name: str, timeout: float = 12.0):
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            print(f"SpatialAgent task timed out: {task_name}")
            return None
        except Exception as e:
            print(f"SpatialAgent task error ({task_name}): {e}")
            return None
