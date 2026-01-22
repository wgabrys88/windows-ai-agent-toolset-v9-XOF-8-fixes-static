import base64
import ctypes
import json
import os
import struct
import sys
import time
import urllib.request
import zlib
import hashlib
import textwrap
from ctypes import wintypes
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional

@dataclass(frozen=True)
class Config:
    lmstudio_endpoint: str = "http://localhost:1234/v1/chat/completions"
    lmstudio_model: str = "qwen3-vl-2b-instruct"
    lmstudio_timeout: int = 240
    lmstudio_temperature: float = 0.5

    planner_max_tokens: int = 2400
    executor_max_tokens: int = 2600


    screen_capture_w: int = 1536
    screen_capture_h: int = 864
    dump_dir: str = "dumps"
    dump_pretty_prompts: bool = True
    dump_pretty_responses: bool = True
    dump_memory_trace: bool = True
    max_steps: int = 50

    ui_settle_delay: float = 0.3
    turn_delay: float = 1.5
    char_input_delay: float = 0.01

    enable_loop_recovery: bool = True
    loop_recovery_cooldown: int = 3

    review_interval: int = 7
    memory_compression_threshold: int = 12

CFG = Config()

class ExecutionLogger:
    def __init__(self, dump_dir: str):
        self.dump_dir = dump_dir
        self.screenshots_dir = os.path.join(dump_dir, "screenshots")
        self.prompts_dir = os.path.join(dump_dir, "prompts")
        self.log_file = os.path.join(dump_dir, "execution_log.txt")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.prompts_dir, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("EXECUTION LOG START\n")
            f.write("=" * 80 + "\n\n")
        self.api_call_counter = 0
        self._system_prompt_seen: dict[str, int] = {}  # hash -> first api_call id
        self._tools_seen: dict[str, int] = {}          # hash -> first api_call id

    def log(self, message: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

    def log_section(self, title: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n\n")

    def log_api_request(self, agent: str, payload: dict[str, Any]) -> None:
        self.api_call_counter += 1
        redacted = self._redact_payload(payload)
        text = self._format_api_request(agent, payload, redacted, self.api_call_counter)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)
        if CFG.dump_pretty_prompts:
            self.dump_pretty_request(agent=agent, payload=payload, redacted=redacted, call_id=self.api_call_counter)

    def log_api_response(self, agent: str, response: dict[str, Any]) -> None:
        text = self._format_api_response(agent, response, self.api_call_counter)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)
        if CFG.dump_pretty_responses:
            self.dump_pretty_response(agent=agent, response=response, call_id=self.api_call_counter)



    def dump_pretty_request(self, agent: str, payload: dict[str, Any], redacted: dict[str, Any], call_id: int) -> None:
        """
        Write a human-readable prompt dump for this API request.
        - Excludes tool schemas (only lists tool names).
        - Avoids duplicating system prompt text; stores system prompt once and references it here.
        """
        try:
            path = os.path.join(self.prompts_dir, f"request_{call_id:04d}_{agent}.txt")

            model = payload.get("model", "?")
            temp = payload.get("temperature", "?")
            max_tokens = payload.get("max_tokens", "?")
            tool_choice = payload.get("tool_choice", "?")

            msgs = payload.get("messages") or []
            sys_prompt = ""
            if msgs and isinstance(msgs, list) and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                sys_prompt = str(msgs[0].get("content") or "")
            sp_hash = self._hash_short(sys_prompt) if sys_prompt else "none"
            sp_path = ""
            if sys_prompt:
                sp_path = os.path.join(self.prompts_dir, f"system_{sp_hash}.txt")
                # Ensure the system prompt is saved once (file-based cache).
                if not os.path.exists(sp_path):
                    with open(sp_path, "w", encoding="utf-8") as pf:
                        pf.write(sys_prompt)

            tool_names: list[str] = []
            tools = payload.get("tools") or []
            if isinstance(tools, list):
                for t in tools:
                    try:
                        tool_names.append(((t.get("function") or {}).get("name")) or "?")
                    except Exception:
                        tool_names.append("?")
            tools_line = ", ".join(tool_names) if tool_names else "<none>"

            # Use redacted messages so base64 images don't explode the file.
            red_msgs = (redacted.get("messages") or []) if isinstance(redacted, dict) else []

            out: list[str] = []
            out.append("=" * 80 + "\n")
            out.append(f"API REQUEST #{call_id} ({agent})\n")
            out.append("=" * 80 + "\n\n")
            out.append(f"Model: {model}\nTemp: {temp}\nMaxTokens: {max_tokens}\nToolChoice: {tool_choice}\n\n")
            if sys_prompt:
                out.append(f"SystemPromptRef: SP_{sp_hash}  ({os.path.basename(sp_path)})\n\n")
            else:
                out.append("SystemPromptRef: <none>\n\n")
            out.append(f"ToolsAvailable: {tools_line}\n\n")
            out.append("MESSAGES\n")
            out.append("-" * 80 + "\n")

            for i, m in enumerate(red_msgs):
                role = (m.get("role") or "?").upper()
                content = m.get("content")
                out.append(f"[{i}] {role}\n")
                out.append("-" * 80 + "\n")

                # We intentionally do NOT re-dump the whole system prompt every time.
                if role == "SYSTEM" and sys_prompt:
                    out.append(f"(system prompt stored in {os.path.basename(sp_path)}; len={len(sys_prompt)})\n\n")
                    continue

                if isinstance(content, str):
                    out.append(content.rstrip() + "\n\n")
                elif isinstance(content, list):
                    for j, part in enumerate(content):
                        ptype = part.get("type", "?")
                        if ptype == "text":
                            t = (part.get("text") or "").rstrip()
                            out.append(f"(part {j}) TEXT\n")
                            out.append(t + "\n\n")
                        elif ptype == "image_url":
                            url = ((part.get("image_url") or {}).get("url")) or ""
                            out.append(f"(part {j}) IMAGE_URL\n{url}\n\n")
                        else:
                            out.append(f"(part {j}) {ptype}\n{str(part)}\n\n")
                else:
                    out.append(str(content) + "\n\n")

            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(out))

            self.log(f"PromptDump: {os.path.basename(path)}")
        except Exception as e:
            self.log(f"PromptDump failed: {e}")

    def dump_pretty_response(self, agent: str, response: dict[str, Any], call_id: int) -> None:
        """
        Write a human-readable response dump for this API call.
        Includes assistant text + tool call name/args (no schemas).
        """
        try:
            path = os.path.join(self.prompts_dir, f"response_{call_id:04d}_{agent}.txt")
            out: list[str] = []
            out.append("=" * 80 + "\n")
            out.append(f"API RESPONSE #{call_id} ({agent})\n")
            out.append("=" * 80 + "\n\n")

            choices = response.get("choices") or []
            if not choices:
                out.append("No choices returned.\n")
            else:
                msg = (choices[0] or {}).get("message") or {}
                content = (msg.get("content") or "").rstrip()
                tool_calls = msg.get("tool_calls") or []
                finish = (choices[0] or {}).get("finish_reason", "")

                out.append(f"FinishReason: {finish}\n\n")
                out.append("ASSISTANT TEXT\n")
                out.append("-" * 80 + "\n")
                out.append((content or "<empty>") + "\n\n")

                out.append("TOOL CALL\n")
                out.append("-" * 80 + "\n")
                if tool_calls:
                    tc0 = tool_calls[0]
                    name = ((tc0.get("function") or {}).get("name")) or "?"
                    raw_args = ((tc0.get("function") or {}).get("arguments"))
                    parsed_args: Any = None
                    if isinstance(raw_args, str):
                        arg_preview = raw_args.strip()
                        try:
                            parsed_args = json.loads(arg_preview) if arg_preview else {}
                        except Exception:
                            parsed_args = None
                    elif isinstance(raw_args, dict):
                        parsed_args = raw_args
                        arg_preview = json.dumps(raw_args, ensure_ascii=False)
                    else:
                        arg_preview = str(raw_args)
                        parsed_args = None

                    def _fmt_obj(o: Any, ind: int = 0) -> str:
                        pad = "  " * ind
                        if isinstance(o, dict):
                            lines: list[str] = []
                            for k, v in o.items():
                                if isinstance(v, (dict, list, tuple)):
                                    lines.append(f"{pad}{k}:")
                                    lines.append(_fmt_obj(v, ind + 1))
                                else:
                                    lines.append(f"{pad}{k}: {v}")
                            return "\n".join(lines) if lines else f"{pad}<empty>"
                        if isinstance(o, (list, tuple)):
                            if not o:
                                return f"{pad}<empty>"
                            lines = []
                            for item in o:
                                if isinstance(item, (dict, list, tuple)):
                                    lines.append(f"{pad}-")
                                    lines.append(_fmt_obj(item, ind + 1))
                                else:
                                    lines.append(f"{pad}- {item}")
                            return "\n".join(lines)
                        return f"{pad}{o}"

                    out.append(f"name: {name}\n\n")
                    out.append("args (pretty):\n")
                    out.append((_fmt_obj(parsed_args, 0) if parsed_args is not None else "<unparsed>") + "\n\n")
                    out.append("args (raw):\n")
                    out.append(arg_preview + "\n")
                else:
                    out.append("<none>\n")

            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(out))

            self.log(f"ResponseDump: {os.path.basename(path)}")
        except Exception as e:
            self.log(f"ResponseDump failed: {e}")

    def log_memory_trace(self, event: str, memory: "AgentMemory") -> None:
        """
        Debug-only: dump full memory state whenever it is read or written.
        """
        if not CFG.dump_memory_trace:
            return
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n" + ("-" * 80) + "\n")
                f.write(f"MEMORY TRACE: {event}\n")
                f.write("-" * 80 + "\n")
                snaps = memory.all_snapshots
                f.write(f"Snapshots: {len(snaps)}\n")
                for i, s in enumerate(snaps, 1):
                    f.write(f"  [{i}] archived_count={s.archived_count}\n")
                    f.write(f"      summary: {s.summary}\n")
                    if s.patterns:
                        f.write(f"      patterns: {s.patterns}\n")
                hist = getattr(memory, "_history", [])
                f.write(f"History: {len(hist)} (active={len(memory.active_history)})\n")
                for rec in hist:
                    f.write(f"  Turn {rec.turn} | tool={rec.tool} | outcome={rec.outcome} | archived={rec.archived} | validated_on={rec.validated_on}\n")
                    if rec.validation_note:
                        f.write(f"    validation_note: {rec.validation_note}\n")
                    f.write("    args:\n")
                    f.write(self._format_tool_args(rec.args))
                    f.write("\n")
                    if rec.result:
                        f.write(f"    result: {rec.result}\n")
                    if rec.screenshot:
                        f.write(f"    screenshot: {rec.screenshot}\n")
        except Exception as e:
            self.log(f"MemoryTrace failed: {e}")
    def log_tool_execution(self, turn: int, tool_name: str, args: Any, action_outcome: str, result: str, prev_outcome: Optional[str] = None) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[TURN {turn}] EXECUTE {tool_name}\n")
            if prev_outcome:
                f.write(f"PrevOutcomeReported: {prev_outcome}\n")
            f.write(f"ActionOutcome: {action_outcome}\n")
            f.write("Args:\n")
            f.write(self._format_tool_args(args))
            f.write("\n")
            f.write(f"Result: {result}\n")

    def log_state_update(self, state_info: dict[str, Any]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\nSTATE UPDATE:\n")
            for k, v in state_info.items():
                f.write(f"  {k}: {v}\n")

    def save_screenshot(self, png: bytes, turn: int) -> str:
        path = os.path.join(self.screenshots_dir, f"turn_{turn:04d}.png")
        with open(path, "wb") as f:
            f.write(png)
        self.log(f"Screenshot: {os.path.basename(path)}")
        return path

    def _hash_short(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]

    def _wrap(self, s: str, width: int = 110, indent: str = "    ") -> str:
        s = (s or "").strip()
        if not s:
            return indent + "<empty>"
        return "\n".join(indent + line for line in textwrap.fill(s, width=width).splitlines())

    def _format_messages(self, messages: list[Any]) -> str:
        lines: list[str] = []
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            content = m.get("content")
            if isinstance(content, str):
                lines.append(f"  [{i}] {role}: text (len={len(content)})")
                snippet = content.replace("\n", " ").strip()[:220]
                if snippet:
                    lines.append(self._wrap(snippet))
            elif isinstance(content, list):
                lines.append(f"  [{i}] {role}: multimodal ({len(content)} parts)")
                for j, part in enumerate(content):
                    ptype = part.get("type", "?")
                    if ptype == "text":
                        t = part.get("text", "")
                        lines.append(f"    - part[{j}] text (len={len(t)})")
                        snippet = t.replace("\n", " ").strip()[:220]
                        if snippet:
                            lines.append(self._wrap(snippet, indent="      "))
                    elif ptype == "image_url":
                        url = ((part.get("image_url") or {}).get("url")) or ""
                        lines.append(f"    - part[{j}] image_url {url}")
                    else:
                        lines.append(f"    - part[{j}] {ptype}: {str(part)[:160]}")
            else:
                lines.append(f"  [{i}] {role}: <unknown content type>")
        return "\n".join(lines) + ("\n" if lines else "")

    def _format_api_request(self, agent: str, payload: dict[str, Any], redacted: dict[str, Any], call_id: int) -> str:
        model = payload.get("model", "?")
        temp = payload.get("temperature", "?")
        max_tokens = payload.get("max_tokens", "?")
        tool_choice = payload.get("tool_choice", "?")

        sys_prompt = ""
        msgs = payload.get("messages") or []
        if msgs and isinstance(msgs, list) and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            sys_prompt = str(msgs[0].get("content") or "")
        sp_hash = self._hash_short(sys_prompt) if sys_prompt else "none"
        if sys_prompt:
            if sp_hash not in self._system_prompt_seen:
                self._system_prompt_seen[sp_hash] = call_id
                with open(os.path.join(self.prompts_dir, f"system_{sp_hash}.txt"), "w", encoding="utf-8") as pf:
                    pf.write(sys_prompt)
                sp_note = f"SystemPrompt: saved (hash=SP_{sp_hash})"
            else:
                sp_note = f"SystemPrompt: cached (hash=SP_{sp_hash}, first_seen=# {self._system_prompt_seen[sp_hash]})"
        else:
            sp_note = "SystemPrompt: <none>"

        tools_red = redacted.get("tools")
        tools_sig = str(tools_red or "")
        if tools_sig:
            tools_note = f"ToolsAvailable: {tools_red}"
        else:
            tools_note = "ToolsAvailable: <none>"

        out = []
        out.append(f"\n--- API REQUEST #{call_id} ({agent}) ---\n")
        out.append(f"Model: {model}  Temp:{temp}  MaxTokens:{max_tokens}  ToolChoice:{tool_choice}\n")
        out.append(f"{sp_note}\n")
        out.append(f"{tools_note}\n")
        out.append("Messages:\n")
        out.append(self._format_messages(redacted.get("messages") or []))
        return "".join(out)

    def _format_api_response(self, agent: str, response: dict[str, Any], call_id: int) -> str:
        out = []
        out.append(f"\n--- API RESPONSE #{call_id} ({agent}) ---\n")
        choices = response.get("choices") or []
        if not choices:
            out.append("No choices returned.\n")
            return "".join(out)

        msg = (choices[0] or {}).get("message") or {}
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        finish = (choices[0] or {}).get("finish_reason", "")

        out.append(f"FinishReason: {finish}\n")
        out.append(f"AssistantText: (len={len(content)})\n")
        if content:
            out.append(self._wrap(content) + "\n")

        if tool_calls:
            tc0 = tool_calls[0]
            name = ((tc0.get("function") or {}).get("name")) or "?"
            raw_args = ((tc0.get("function") or {}).get("arguments"))
            if isinstance(raw_args, str):
                arg_preview = raw_args.strip()
            else:
                arg_preview = str(raw_args)
            out.append("ToolCall:\n")
            out.append(f"  name: {name}\n")
            out.append(f"  args: (len={len(arg_preview)})\n")
            out.append(self._wrap(arg_preview[:900], indent="    ") + ("\n" if len(arg_preview) > 900 else "\n"))
        else:
            out.append("ToolCall: <none>\n")

        return "".join(out)

    def _redact_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, dict):
            out = {}
            for k, v in payload.items():
                if k == "url" and isinstance(v, str) and v.startswith("data:image/png;base64,"):
                    b64 = v.split(",", 1)[1]
                    out[k] = f"<base64_png len={len(b64)}>"
                elif k == "tools" and isinstance(v, list):
                    tool_names = [t.get('function', {}).get('name', '?') for t in v]
                    out[k] = f"<{len(v)} tools: {', '.join(tool_names)}>"
                else:
                    out[k] = self._redact_payload(v)
            return out
        if isinstance(payload, list):
            return [self._redact_payload(x) for x in payload]
        return payload

    def _format_tool_args(self, args: Any) -> str:
        try:
            if isinstance(args, ClickArgs):
                markers = "\n".join("    - " + m for m in (args.change_markers or ()))
                markers = markers if markers else "    <none>"
                return (
                    f"  element: {args.element_name}\n"
                    f"  position: [{args.position.x:.0f},{args.position.y:.0f}]\n"
                    f"  expected_visible_change:\n{self._wrap(args.expected_visible_change)}\n"
                    f"  change_markers:\n{markers}\n"
                    f"  visual_justification:\n{self._wrap(args.visual_justification)}\n"
                    f"  reasoning:\n{self._wrap(args.reasoning)}\n"
                )
            if isinstance(args, DragArgs):
                markers = "\n".join("    - " + m for m in (args.change_markers or ()))
                markers = markers if markers else "    <none>"
                return (
                    f"  element: {args.element_name}\n"
                    f"  start: [{args.start.x:.0f},{args.start.y:.0f}]  end: [{args.end.x:.0f},{args.end.y:.0f}]\n"
                    f"  expected_visible_change:\n{self._wrap(args.expected_visible_change)}\n"
                    f"  change_markers:\n{markers}\n"
                    f"  visual_justification:\n{self._wrap(args.visual_justification)}\n"
                    f"  reasoning:\n{self._wrap(args.reasoning)}\n"
                )
            if isinstance(args, TypeTextArgs):
                return (
                    f"  text: {args.text}\n"
                    f"  reasoning:\n{self._wrap(args.reasoning)}\n"
                )
            if isinstance(args, PressKeyArgs):
                return (
                    f"  key: {args.key}\n"
                    f"  reasoning:\n{self._wrap(args.reasoning)}\n"
                )
            if isinstance(args, ScrollArgs):
                return f"  reasoning:\n{self._wrap(args.reasoning)}\n"
            if isinstance(args, ReportProgressArgs):
                return (
                    f"  goal: {args.goal_identifier}\n"
                    f"  status: {args.completion_status}\n"
                    f"  evidence:\n{self._wrap(args.evidence)}\n"
                    f"  prev_outcome: {args.outcome_status}\n"
                    f"  prev_reasoning:\n{self._wrap(args.reasoning)}\n"
                )
            if isinstance(args, ReportCompletionArgs):
                return (
                    f"  evidence:\n{self._wrap(args.evidence)}\n"
                    f"  prev_outcome: {args.outcome_status}\n"
                    f"  prev_reasoning:\n{self._wrap(args.reasoning)}\n"
                )
            if isinstance(args, UpdatePlanArgs):
                return (
                    f"  reasoning:\n{self._wrap(args.reasoning)}\n"
                    f"  instructions:\n{self._wrap(args.instructions)}\n"
                )
            if isinstance(args, ArchiveHistoryArgs):
                return (
                    f"  summary:\n{self._wrap(args.summary)}\n"
                    f"  patterns:\n{self._wrap(args.patterns_detected)}\n"
                    f"  archived_turns: {list(args.archived_turns)}\n"
                )
        except Exception:
            pass
        return "  " + self._wrap(str(args), indent="  ")

LOGGER = ExecutionLogger(CFG.dump_dir)

def log_api_call(agent_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            payload = kwargs.get('payload') or (args[0] if args else None)
            if payload:
                LOGGER.log_api_request(agent_name, payload)
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                LOGGER.log_api_response(agent_name, result)
            return result
        return wrapper
    return decorator

for attr in ["HCURSOR", "HICON", "HBITMAP", "HGDIOBJ", "HBRUSH", "HDC"]:
    if not hasattr(wintypes, attr):
        setattr(wintypes, attr, wintypes.HANDLE)
if not hasattr(wintypes, "ULONG_PTR"):
    wintypes.ULONG_PTR = ctypes.c_size_t

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
SM_CXSCREEN, SM_CYSCREEN = 0, 1
CURSOR_SHOWING, DI_NORMAL = 0x00000001, 0x0003
BI_RGB, DIB_RGB_COLORS = 0, 0
HALFTONE, SRCCOPY = 4, 0x00CC0020
INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0002, 0x0004
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_WHEEL = 0x0800

VK_MAP = {
    "enter": 0x0D, "tab": 0x09, "escape": 0x1B, "esc": 0x1B, "windows": 0x5B, "win": 0x5B,
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10, "backspace": 0x08, "delete": 0x2E, "space": 0x20,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}
for i in range(ord('a'), ord('z') + 1):
    VK_MAP[chr(i)] = ord(chr(i).upper())
for i in range(10):
    VK_MAP[str(i)] = 0x30 + i
for i in range(1, 13):
    VK_MAP[f"f{i}"] = 0x70 + (i - 1)

class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

class CURSORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("hCursor", wintypes.HCURSOR), ("ptScreenPos", POINT)]

class ICONINFO(ctypes.Structure):
    _fields_ = [("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD),
                ("yHotspot", wintypes.DWORD), ("hbmMask", wintypes.HBITMAP),
                ("hbmColor", wintypes.HBITMAP)]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR)]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR)]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

class INPUT_I(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_I)]

user32.GetSystemMetrics.argtypes = [wintypes.INT]
user32.GetSystemMetrics.restype = wintypes.INT
user32.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
user32.GetCursorInfo.restype = wintypes.BOOL
user32.GetIconInfo.argtypes = [wintypes.HICON, ctypes.POINTER(ICONINFO)]
user32.GetIconInfo.restype = wintypes.BOOL
user32.DrawIconEx.argtypes = [wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.HICON,
                              wintypes.INT, wintypes.INT, wintypes.UINT, wintypes.HBRUSH, wintypes.UINT]
user32.DrawIconEx.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = wintypes.INT
user32.SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
user32.SetCursorPos.restype = wintypes.BOOL
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
                                    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.StretchBlt.argtypes = [wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
                             wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.DWORD]
gdi32.StretchBlt.restype = wintypes.BOOL
gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, wintypes.INT]
gdi32.SetStretchBltMode.restype = wintypes.INT
gdi32.SetBrushOrgEx.argtypes = [wintypes.HDC, wintypes.INT, wintypes.INT, ctypes.POINTER(POINT)]
gdi32.SetBrushOrgEx.restype = wintypes.BOOL

def init_dpi() -> None:
    user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)

def get_screen_size() -> tuple[int, int]:
    w = user32.GetSystemMetrics(SM_CXSCREEN)
    h = user32.GetSystemMetrics(SM_CYSCREEN)
    return (w if w > 0 else 1920, h if h > 0 else 1080)

def png_pack(tag: bytes, data: bytes) -> bytes:
    chunk = tag + data
    return struct.pack("!I", len(data)) + chunk + struct.pack("!I", zlib.crc32(chunk) & 0xFFFFFFFF)

def rgb_to_png(rgb: bytes, w: int, h: int) -> bytes:
    raw = bytearray(b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h)))
    compressed = zlib.compress(bytes(raw), level=6)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(png_pack(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    png.extend(png_pack(b"IDAT", compressed))
    png.extend(png_pack(b"IEND", b""))
    return bytes(png)

def draw_cursor(hdc_mem: int, sw: int, sh: int, dw: int, dh: int) -> None:
    ci = CURSORINFO(cbSize=ctypes.sizeof(CURSORINFO))
    if not user32.GetCursorInfo(ctypes.byref(ci)) or not (ci.flags & CURSOR_SHOWING):
        return
    ii = ICONINFO()
    if not user32.GetIconInfo(ci.hCursor, ctypes.byref(ii)):
        return
    try:
        cx = int(ci.ptScreenPos.x) - int(ii.xHotspot)
        cy = int(ci.ptScreenPos.y) - int(ii.yHotspot)
        dx = int(round(cx * (dw / float(sw))))
        dy = int(round(cy * (dh / float(sh))))
        user32.DrawIconEx(hdc_mem, dx, dy, ci.hCursor, 0, 0, 0, None, DI_NORMAL)
    finally:
        if ii.hbmMask:
            gdi32.DeleteObject(ii.hbmMask)
        if ii.hbmColor:
            gdi32.DeleteObject(ii.hbmColor)

def capture_png(tw: int, th: int) -> tuple[bytes, int, int]:
    sw, sh = get_screen_size()
    hdc_scr = user32.GetDC(None)
    if not hdc_scr:
        raise RuntimeError("GetDC failed")
    hdc_mem = gdi32.CreateCompatibleDC(hdc_scr)
    if not hdc_mem:
        user32.ReleaseDC(None, hdc_scr)
        raise RuntimeError("CreateCompatibleDC failed")

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth, bmi.bmiHeader.biHeight = tw, -th
    bmi.bmiHeader.biPlanes, bmi.bmiHeader.biBitCount = 1, 32
    bmi.bmiHeader.biCompression = BI_RGB
    bits = ctypes.c_void_p()
    hbm = gdi32.CreateDIBSection(hdc_scr, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
    if not hbm or not bits:
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_scr)
        raise RuntimeError("CreateDIBSection failed")

    old = gdi32.SelectObject(hdc_mem, hbm)
    gdi32.SetStretchBltMode(hdc_mem, HALFTONE)
    gdi32.SetBrushOrgEx(hdc_mem, 0, 0, None)
    if not gdi32.StretchBlt(hdc_mem, 0, 0, tw, th, hdc_scr, 0, 0, sw, sh, SRCCOPY):
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_scr)
        raise RuntimeError("StretchBlt failed")

    draw_cursor(hdc_mem, sw, sh, tw, th)
    raw = bytes((ctypes.c_ubyte * (tw * th * 4)).from_address(bits.value))
    gdi32.SelectObject(hdc_mem, old)
    gdi32.DeleteObject(hbm)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(None, hdc_scr)

    rgb = bytearray(tw * th * 3)
    for i in range(tw * th):
        rgb[i * 3:i * 3 + 3] = [raw[i * 4 + 2], raw[i * 4 + 1], raw[i * 4 + 0]]
    return rgb_to_png(bytes(rgb), tw, th), sw, sh

def send_input_events(events: tuple[INPUT, ...]) -> None:
    arr = (INPUT * len(events))(*events)
    if user32.SendInput(len(events), arr, ctypes.sizeof(INPUT)) != len(events):
        raise RuntimeError("SendInput failed")

def mouse_move(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))
    time.sleep(CFG.ui_settle_delay)

def mouse_click(button: str = "left") -> None:
    if button == "left":
        down_flag, up_flag = MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
    elif button == "right":
        down_flag, up_flag = MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
    else:
        raise ValueError(f"Unknown button: {button}")

    send_input_events((
        INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=down_flag, time=0, dwExtraInfo=0))),
        INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=up_flag, time=0, dwExtraInfo=0)))
    ))
    time.sleep(CFG.ui_settle_delay)

def mouse_double_click() -> None:
    mouse_click("left")
    mouse_click("left")

def mouse_drag(x1: int, y1: int, x2: int, y2: int) -> None:
    mouse_move(x1, y1)
    send_input_events((INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),))
    time.sleep(CFG.ui_settle_delay)

    steps = 15
    for i in range(1, steps + 1):
        t = i / float(steps)
        mouse_move(int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t))

    send_input_events((INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),))
    time.sleep(CFG.ui_settle_delay)

def mouse_scroll(direction: int) -> None:
    delta = 120 if direction > 0 else -120
    send_input_events((INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(dx=0, dy=0, mouseData=delta, dwFlags=MOUSEEVENTF_WHEEL, time=0, dwExtraInfo=0))),))
    time.sleep(CFG.ui_settle_delay)

def keyboard_type_text(text: str) -> None:
    all_events = []
    for ch in text:
        code = ord(ch)
        all_events.append(INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0))))
        all_events.append(INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))))
        if len(all_events) >= 100:
            send_input_events(tuple(all_events))
            all_events = []
            time.sleep(CFG.char_input_delay)
    if all_events:
        send_input_events(tuple(all_events))
    time.sleep(CFG.ui_settle_delay)

def keyboard_press_keys(key_combo: str) -> None:
    parts = tuple(p.strip() for p in key_combo.strip().lower().split("+") if p.strip())
    vks = tuple(VK_MAP[p] for p in parts)

    events = tuple(INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=0))) for vk in vks)
    events += tuple(INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))) for vk in reversed(vks))

    send_input_events(events)
    time.sleep(CFG.ui_settle_delay)

@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float

    def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int]:
        xn = max(0.0, min(1000.0, self.x))
        yn = max(0.0, min(1000.0, self.y))
        px = min(int(round((xn / 1000.0) * screen_width)), screen_width - 1)
        py = min(int(round((yn / 1000.0) * screen_height)), screen_height - 1)
        return (px, py)

@dataclass(frozen=True)
class ClickArgs:
    outcome_status: str
    reasoning: str
    element_name: str
    position: Coordinate
    visual_justification: str = ""
    expected_visible_change: str = ""
    change_markers: tuple[str, ...] = ()

@dataclass(frozen=True)
class DragArgs:
    outcome_status: str
    reasoning: str
    element_name: str
    start: Coordinate
    end: Coordinate
    visual_justification: str = ""
    expected_visible_change: str = ""
    change_markers: tuple[str, ...] = ()

@dataclass(frozen=True)
class TypeTextArgs:
    outcome_status: str
    reasoning: str
    text: str

@dataclass(frozen=True)
class PressKeyArgs:
    outcome_status: str
    reasoning: str
    key: str

@dataclass(frozen=True)
class ScrollArgs:
    outcome_status: str
    reasoning: str

@dataclass(frozen=True)
class ReportCompletionArgs:
    evidence: str
    outcome_status: str = "UNKNOWN"
    reasoning: str = ""

@dataclass(frozen=True)
class ReportProgressArgs:
    goal_identifier: str
    completion_status: str
    evidence: str
    outcome_status: str = "UNKNOWN"
    reasoning: str = ""

@dataclass(frozen=True)
class ArchiveHistoryArgs:
    summary: str
    patterns_detected: str
    archived_turns: tuple[int, ...]

@dataclass(frozen=True)
class UpdatePlanArgs:
    instructions: str
    reasoning: str

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Any

    @staticmethod
    def parse(tool_call_dict: dict[str, Any]) -> 'ToolCall':
        name = tool_call_dict["function"]["name"]
        raw_args = tool_call_dict["function"].get("arguments")
        if isinstance(raw_args, str):
            args_dict = json.loads(raw_args) if raw_args.strip() else {}
        elif isinstance(raw_args, dict):
            args_dict = raw_args
        else:
            args_dict = {}

        parsers = {
            "click_screen_element": lambda d: ClickArgs(
                outcome_status=d.get("outcome_status", "UNKNOWN"),
                reasoning=d.get("reasoning", ""),
                element_name=d.get("element_name", ""),
                position=Coordinate(float(d.get("position", [0, 0])[0]), float(d.get("position", [0, 0])[1])),
                visual_justification=d.get("visual_justification", ""),
                expected_visible_change=d.get("expected_visible_change", ""),
                change_markers=tuple(d.get("change_markers", []) or ())
            ),
            "double_click_screen_element": lambda d: ClickArgs(
                outcome_status=d.get("outcome_status", "UNKNOWN"),
                reasoning=d.get("reasoning", ""),
                element_name=d.get("element_name", ""),
                position=Coordinate(float(d.get("position", [0, 0])[0]), float(d.get("position", [0, 0])[1])),
                visual_justification=d.get("visual_justification", ""),
                expected_visible_change=d.get("expected_visible_change", ""),
                change_markers=tuple(d.get("change_markers", []) or ())
            ),
            "right_click_screen_element": lambda d: ClickArgs(
                outcome_status=d.get("outcome_status", "UNKNOWN"),
                reasoning=d.get("reasoning", ""),
                element_name=d.get("element_name", ""),
                position=Coordinate(float(d.get("position", [0, 0])[0]), float(d.get("position", [0, 0])[1])),
                visual_justification=d.get("visual_justification", ""),
                expected_visible_change=d.get("expected_visible_change", ""),
                change_markers=tuple(d.get("change_markers", []) or ())
            ),
            "drag_screen_element": lambda d: DragArgs(
                outcome_status=d.get("outcome_status", "UNKNOWN"),
                reasoning=d.get("reasoning", ""),
                element_name=d.get("element_name", ""),
                start=Coordinate(float(d.get("start", [0, 0])[0]), float(d.get("start", [0, 0])[1])),
                end=Coordinate(float(d.get("end", [0, 0])[0]), float(d.get("end", [0, 0])[1])),
                visual_justification=d.get("visual_justification", ""),
                expected_visible_change=d.get("expected_visible_change", ""),
                change_markers=tuple(d.get("change_markers", []) or ())
            ),
            "type_text_input": lambda d: TypeTextArgs(d.get("outcome_status", "UNKNOWN"), d.get("reasoning", ""), d.get("text", "")),
            "press_keyboard_key": lambda d: PressKeyArgs(d.get("outcome_status", "UNKNOWN"), d.get("reasoning", ""), d.get("key", "")),
            "scroll_screen_down": lambda d: ScrollArgs(d.get("outcome_status", "UNKNOWN"), d.get("reasoning", "")),
            "scroll_screen_up": lambda d: ScrollArgs(d.get("outcome_status", "UNKNOWN"), d.get("reasoning", "")),
            "report_goal_status": lambda d: ReportProgressArgs(
                d.get("goal_identifier", ""),
                d.get("completion_status", ""),
                d.get("evidence", ""),
                d.get("outcome_status", "UNKNOWN"),
                d.get("reasoning", "")
            ),
            "report_mission_complete": lambda d: ReportCompletionArgs(
                d.get("evidence", ""),
                d.get("outcome_status", "UNKNOWN"),
                d.get("reasoning", "")
            ),
            "archive_completed_actions": lambda d: ArchiveHistoryArgs(d.get("summary", ""), d.get("patterns_detected", ""), tuple(d.get("archived_turns", []))),
            "update_execution_plan": lambda d: UpdatePlanArgs(d.get("instructions", ""), d.get("reasoning", "")),
        }

        if name not in parsers:
            raise ValueError(f"Unknown tool: {name}")

        return ToolCall(name=name, arguments=parsers[name](args_dict))

PLANNER_TOOLS = (
    {"type": "function", "function": {"name": "update_execution_plan", "description": "Create or update step-by-step execution plan",
     "parameters": {"type": "object", "properties": {
         "instructions": {"type": "string", "description": "Clear step-by-step instructions for executor"},
         "reasoning": {"type": "string", "description": "Why this plan addresses current situation"}},
         "required": ["instructions", "reasoning"]}}},

    {"type": "function", "function": {"name": "archive_completed_actions", "description": "Compress old action history to save context space",
     "parameters": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "Summary of what was accomplished"},
         "patterns_detected": {"type": "string", "description": "Any repetition patterns or issues found"},
         "archived_turns": {"type": "array", "items": {"type": "integer"}, "description": "Turn numbers to archive"}},
         "required": ["summary", "patterns_detected", "archived_turns"]}}},
)

EXECUTOR_TOOLS = (
    {"type": "function", "function": {"name": "report_mission_complete", "description": "Declare entire mission finished - terminal action",
     "parameters": {"type": "object", "properties": {
         "evidence": {"type": "string", "description": "Detailed proof showing task is fully complete"},
         "outcome_status": {"type": "string", "description": "Assessment of the PREVIOUS action outcome if applicable", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string", "description": "PrevCheck/NowAction/Expected style note if applicable"}},
         "required": ["evidence"]}}},

    {"type": "function", "function": {"name": "report_goal_status", "description": "Report progress on current goal",
     "parameters": {"type": "object", "properties": {
         "goal_identifier": {"type": "string", "description": "Which goal from plan this status is for"},
         "completion_status": {"type": "string", "description": "Must be DONE, IN_PROGRESS, or BLOCKED"},
         "evidence": {"type": "string", "description": "What you see in screenshot that proves this status"},
         "outcome_status": {"type": "string", "description": "Assessment of the PREVIOUS action outcome if applicable", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string", "description": "PrevCheck/NowAction/Expected style note if applicable"}},
         "required": ["goal_identifier", "completion_status", "evidence"]}}},

{
  "type": "function",
  "function": {
    "name": "click_screen_element",
    "description": "Click visible UI element with left mouse button",
    "parameters": {
      "type": "object",
      "properties": {
        "outcome_status": {
          "type": "string",
          "description": "PROGRESS if UI changed as expected, NO_CHANGE if UI looks the same, BLOCKED if tried multiple times with no progress",
          "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]
        },
        "reasoning": {
          "type": "string",
          "description": "PrevCheck/NowAction/Expected (concise). Must reference change_markers for previous outcome validation."
        },
        "visual_justification": {
          "type": "string",
          "description": "Long, concrete description of what you see and why this is the correct target. Include nearby anchors, colors/shapes/icons/text, and where on screen it is."
        },
        "expected_visible_change": {
          "type": "string",
          "description": "Primary visible change expected after the click (what should appear/disappear/move)."
        },
        "change_markers": {
          "type": "array",
          "items": {"type": "string"},
          "description": "3-7 specific on-screen markers (anchors + expected change) to verify success next turn."
        },
        "element_name": {
          "type": "string",
          "description": "Descriptive name of what you are clicking"
        },
        "position": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2,
          "description": "Grid coordinates [x, y] where 0 to 1000"
        }
      },
      "required": [
        "outcome_status",
        "reasoning",
        "visual_justification",
        "expected_visible_change",
        "change_markers",
        "element_name",
        "position"
      ]
    }
  }
},

{
  "type": "function",
  "function": {
    "name": "double_click_screen_element",
    "description": "Double-click UI element with left mouse button",
    "parameters": {
      "type": "object",
      "properties": {
        "outcome_status": {
          "type": "string",
          "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]
        },
        "reasoning": {
          "type": "string",
          "description": "PrevCheck/NowAction/Expected (concise). Must reference change_markers for previous outcome validation."
        },
        "visual_justification": {
          "type": "string",
          "description": "Long, concrete description of what you see and why this is the correct target. Include nearby anchors and where on screen it is."
        },
        "expected_visible_change": {
          "type": "string",
          "description": "Primary visible change expected after the double-click."
        },
        "change_markers": {
          "type": "array",
          "items": {"type": "string"},
          "description": "3-7 specific on-screen markers (anchors + expected change) to verify success next turn."
        },
        "element_name": {"type": "string"},
        "position": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2
        }
      },
      "required": [
        "outcome_status",
        "reasoning",
        "visual_justification",
        "expected_visible_change",
        "change_markers",
        "element_name",
        "position"
      ]
    }
  }
},

{
  "type": "function",
  "function": {
    "name": "right_click_screen_element",
    "description": "Right-click UI element to open context menu",
    "parameters": {
      "type": "object",
      "properties": {
        "outcome_status": {
          "type": "string",
          "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]
        },
        "reasoning": {
          "type": "string",
          "description": "PrevCheck/NowAction/Expected (concise). Must reference change_markers for previous outcome validation."
        },
        "visual_justification": {
          "type": "string",
          "description": "Long, concrete description of what you see and why this is the correct target. Include nearby anchors and where on screen it is."
        },
        "expected_visible_change": {
          "type": "string",
          "description": "Primary visible change expected after right-click (e.g., context menu appears)."
        },
        "change_markers": {
          "type": "array",
          "items": {"type": "string"},
          "description": "3-7 specific on-screen markers (anchors + expected change) to verify success next turn."
        },
        "element_name": {"type": "string"},
        "position": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2
        }
      },
      "required": [
        "outcome_status",
        "reasoning",
        "visual_justification",
        "expected_visible_change",
        "change_markers",
        "element_name",
        "position"
      ]
    }
  }
},

    {"type": "function", "function": {"name": "drag_screen_element", "description": "Drag element from start position to end position",
     "parameters": {"type": "object", "properties": {
         "outcome_status": {"type": "string", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string", "description": "PrevCheck/NowAction/Expected (concise). Must reference change_markers for previous outcome validation."},
         "visual_justification": {"type": "string", "description": "Long, concrete description of what you see and why this is the correct element to drag. Include anchors and screen location."},
         "expected_visible_change": {"type": "string", "description": "Primary visible change expected after the drag (e.g., slider moves, window repositions)."},
         "change_markers": {"type": "array", "items": {"type": "string"}, "description": "3-7 specific on-screen markers (anchors + expected change) to verify success next turn."},
         "element_name": {"type": "string"},
         "start": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
         "end": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
         "required": ["outcome_status", "reasoning", "visual_justification", "expected_visible_change", "change_markers", "element_name", "start", "end"]}}},

    {"type": "function", "function": {"name": "type_text_input", "description": "Type text into currently focused input field",
     "parameters": {"type": "object", "properties": {
         "outcome_status": {"type": "string", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string", "description": "Explain why typing this text achieves current goal"},
         "text": {"type": "string", "description": "Exact text to type including spaces and punctuation"}},
         "required": ["outcome_status", "reasoning", "text"]}}},

    {"type": "function", "function": {"name": "press_keyboard_key", "description": "Press single key or key combination like enter or ctrl+c",
     "parameters": {"type": "object", "properties": {
         "outcome_status": {"type": "string", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string"},
         "key": {"type": "string", "description": "Key name or combo separated by plus sign"}},
         "required": ["outcome_status", "reasoning", "key"]}}},

    {"type": "function", "function": {"name": "scroll_screen_down", "description": "Scroll current window content downward",
     "parameters": {"type": "object", "properties": {
         "outcome_status": {"type": "string", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string", "description": "Why scrolling down helps find target or advance goal"}},
         "required": ["outcome_status", "reasoning"]}}},

    {"type": "function", "function": {"name": "scroll_screen_up", "description": "Scroll current window content upward",
     "parameters": {"type": "object", "properties": {
         "outcome_status": {"type": "string", "enum": ["PROGRESS", "NO_CHANGE", "BLOCKED"]},
         "reasoning": {"type": "string"}},
         "required": ["outcome_status", "reasoning"]}}},

)

@dataclass
class ActionRecord:
    turn: int
    tool: str
    args: Any
    outcome: str
    result: str
    screenshot: str = ""
    archived: bool = False
    validated_on: int = 0
    validation_note: str = ""

@dataclass(frozen=True)
class MemorySnapshot:
    summary: str
    patterns: str
    archived_count: int

class AgentMemory:
    def __init__(self):
        self._history: list[ActionRecord] = []
        self._snapshots: list[MemorySnapshot] = []
        self.last_recovery_turn: int = -10000

    def add_action(self, record: ActionRecord) -> None:
        self._history.append(record)
        LOGGER.log_memory_trace(f"WRITE add_action(turn={record.turn}, tool={record.tool}, outcome={record.outcome})", self)

    @property
    def active_history(self) -> tuple[ActionRecord, ...]:
        return tuple(r for r in self._history if not r.archived)

    @property
    def all_snapshots(self) -> tuple[MemorySnapshot, ...]:
        return tuple(self._snapshots)

    def apply_compression(self, summary: str, patterns: str, archived_turns: tuple[int, ...]) -> None:
        for turn in archived_turns:
            for rec in self._history:
                if rec.turn == turn:
                    rec.archived = True

        self._snapshots.append(MemorySnapshot(
            summary=summary,
            patterns=patterns,
            archived_count=len(archived_turns)
        ))

        LOGGER.log(f"Memory compressed: {len(archived_turns)} actions archived")
        LOGGER.log_memory_trace("WRITE apply_compression", self)

    def get_context_for_llm(self) -> str:
        LOGGER.log_memory_trace("READ get_context_for_llm", self)
        lines = []

        if self._snapshots:
            lines.append("COMPLETED_WORK:")
            for i, snap in enumerate(self._snapshots, 1):
                lines.append(f"  Archive_{i}: {snap.summary[:180]}")
                if snap.patterns:
                    lines.append(f"    Patterns: {snap.patterns[:120]}")
            lines.append("")

        active = self.active_history
        if active:
            lines.append(f"RECENT_ACTIONS (count={len(active)}):")
            for rec in active[-8:]:
                action_summary = self._format_action_summary(rec)
                lines.append(f"  Turn_{rec.turn}: {rec.tool} [{rec.outcome}] {action_summary}")

        return "\n".join(lines)

    def _format_action_summary(self, rec: ActionRecord) -> str:
        if isinstance(rec.args, ClickArgs):
            exp = (rec.args.expected_visible_change or "").strip()
            exp = (exp[:70] + "...") if len(exp) > 70 else exp
            return f"@[{rec.args.position.x:.0f},{rec.args.position.y:.0f}] {rec.args.element_name}" + (f" -> {exp}" if exp else "")
        elif isinstance(rec.args, DragArgs):
            exp = (rec.args.expected_visible_change or "").strip()
            exp = (exp[:70] + "...") if len(exp) > 70 else exp
            base = f"from [{rec.args.start.x:.0f},{rec.args.start.y:.0f}] to [{rec.args.end.x:.0f},{rec.args.end.y:.0f}]"
            return base + (f" -> {exp}" if exp else "")
        elif isinstance(rec.args, TypeTextArgs):
            return f"'{rec.args.text[:30]}'"
        elif isinstance(rec.args, PressKeyArgs):
            return f"key={rec.args.key}"
        elif isinstance(rec.args, (ScrollArgs, ReportProgressArgs)):
            return rec.result[:40]
        return rec.result[:40]

    def get_last_action_for_validation(self) -> Optional[ActionRecord]:
        LOGGER.log_memory_trace("READ get_last_action_for_validation", self)
        active = self.active_history
        return active[-1] if active else None

    def validate_last_pending(self, current_turn: int, outcome: str, validation_note: str = "") -> Optional[ActionRecord]:
        """Mark the most recent pending action as validated, and trace memory mutation."""
        last = self.get_last_action_for_validation()
        if not last:
            return None
        if (last.outcome or "").strip().upper() != "PENDING":
            return None
        last.outcome = (outcome or "").strip().upper() or "UNKNOWN"
        last.validated_on = current_turn
        if validation_note:
            last.validation_note = validation_note[:500]
        LOGGER.log_memory_trace(f"WRITE validate_last_pending(turn={last.turn} => {last.outcome}, validated_on={current_turn})", self)
        return last

    @property
    def needs_compression(self) -> bool:
        return len(self.active_history) >= CFG.memory_compression_threshold

    def mark_recovery(self, turn: int) -> None:
        self.last_recovery_turn = turn
        LOGGER.log_memory_trace(f"WRITE mark_recovery(turn={turn})", self)

    def should_recover(self, current_turn: int) -> bool:
        return (current_turn - self.last_recovery_turn) > CFG.loop_recovery_cooldown

    def recent_outcomes(self, n: int = 3) -> tuple[str, ...]:
        LOGGER.log_memory_trace(f"READ recent_outcomes(n={n})", self)
        vals: list[str] = []
        for rec in reversed(self.active_history):
            o = (rec.outcome or "").strip().upper()
            if not o or o == "PENDING":
                continue
            vals.append(o)
            if len(vals) >= n:
                break
        return tuple(reversed(vals))

    # Tiny patch: repeated-coordinate click detector (3 consecutive click-like actions)
    def detect_repeated_clicks(self, n: int = 3, tol: float = 10.0) -> Optional[str]:
        LOGGER.log_memory_trace(f"READ detect_repeated_clicks(n={n}, tol={tol})", self)
        active = list(self.active_history)
        if len(active) < n:
            return None

        last_n = active[-n:]
        click_tools = {"click_screen_element", "double_click_screen_element", "right_click_screen_element"}
        coords: list[tuple[float, float]] = []
        names: list[str] = []
        tools: list[str] = []

        for rec in last_n:
            if rec.tool not in click_tools:
                return None
            if not isinstance(rec.args, ClickArgs):
                return None
            coords.append((rec.args.position.x, rec.args.position.y))
            names.append((rec.args.element_name or "").strip().lower())
            tools.append(rec.tool)

        if any(not nm for nm in names):
            return None
        if len(set(names)) != 1:
            return None
        if len(set(tools)) != 1:
            return None

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if (max(xs) - min(xs)) <= tol and (max(ys) - min(ys)) <= tol:
            return (
                f"Repeated-click pattern detected: tool={tools[0]} element='{names[0]}' "
                f"positions={[(round(x), round(y)) for (x, y) in coords]}"
            )
        return None

@dataclass
class AgentState:
    task: str
    screenshot: Optional[bytes] = None
    screen_dims: tuple[int, int] = (1920, 1080)
    turn: int = 0
    mode: str = "EXECUTION"

    memory: AgentMemory = field(default_factory=AgentMemory)
    plan: str = ""
    execution_instructions: str = ""
    needs_review: bool = False

    def increment_turn(self) -> None:
        self.turn += 1

    def update_screenshot(self, png: bytes) -> None:
        self.screenshot = png

    def get_context(self) -> str:
        lines = [f"TASK: {self.task}\n"]
        if self.plan:
            excerpt = self.plan[:400] + "..." if len(self.plan) > 400 else self.plan
            lines.append(f"CURRENT_PLAN:\n{excerpt}\n")
        lines.append(f"OPERATING_MODE: {self.mode}\n")
        lines.append(f"CURRENT_TURN: {self.turn}\n\n")
        lines.append(self.memory.get_context_for_llm())
        return "\n".join(lines)

@log_api_call("API")
def post_json(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(CFG.lmstudio_endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=CFG.lmstudio_timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

PLANNER_PROMPT = """You are the TASK PLANNER for a computer-control agent.

TASK:
{task}

{plan_section}

MODE: PLANNING

ROLE
- You do NOT execute UI actions.
- You produce a short plan + precise next instruction for the Executor.
- Treat the screenshot as ground truth. If history says PROGRESS but the screen looks unchanged, assume NO_CHANGE.

OUTPUT (MANDATORY)
- Use update_execution_plan(instructions, reasoning)
- Use archive_completed_actions only when active history reaches {threshold}+ items

INSTRUCTIONS FORMAT (required)
1) CURRENT OBJECTIVE (1 sentence)
2) GOALS (3–7 max), each with:
   - Goal:
   - Success check (must be visible)
3) NEXT ACTION (exactly 1 Executor action for the next turn):
   - Action:
   - Target cues (how to find it on-screen):
   - Expected visible result:
4) IF THAT FAILS (2 fallbacks max; different route or action type)

LOOP RULES
- If you see repeated NO_CHANGE/BLOCKED or repeated coordinates, change strategy (not micro-adjust clicks).
- Prefer stable anchors: window titles, labeled buttons, tabs, nav bars, address bar.
"""

EXECUTOR_PROMPT = """You are the TASK EXECUTOR for a computer-control agent.

TASK:
{task}

MODE: EXECUTION

PLANNER INSTRUCTIONS:
{instructions}

PREVIOUS ACTION:
{previous_action}

HARD CONSTRAINTS
- Call EXACTLY ONE tool per turn.
- Ground actions in what is visible on the screenshot.

CRITICAL SEMANTICS
- outcome_status (inside the tool you call) MUST describe the outcome of the PREVIOUS turn’s action,
  as observed in the CURRENT screenshot.
  * PROGRESS: clear visible change happened as expected
  * NO_CHANGE: screen essentially unchanged / expected change not visible
  * BLOCKED: repeated attempts or a modal/error prevents progress
- If unsure, choose NO_CHANGE (do NOT guess PROGRESS).

REQUIRED REASONING FORMAT (put this in the tool's 'reasoning' field)
PrevCheck: <what changed or didn't change + evidence>
NowAction: <what you will do now + where + why>
Expected: <what should visibly change if it works>

REQUIRED TOOL FIELDS
- reasoning (concise, required; put this in the tool's 'reasoning' field):
  PrevCheck: <validate previous action using PREVIOUS ACTION markers + evidence>
  NowAction: <what you will do now + where + why>
  Expected: <what should visibly change if it works>
- visual_justification (long, required): describe the target + surrounding anchors so a human could find it again.
- expected_visible_change (required): 1–2 sentences describing the main visible change you expect after THIS action.
- change_markers (required): 3–7 marker strings used to validate the outcome next turn.
  Each marker MUST include (a) anchor description, (b) where on screen, and (c) what should change.

ANTI-LOOP RULES
- Do not repeat the exact same click after NO_CHANGE.
- After 2 NO_CHANGE attempts on similar targets, change action type or route.
- If you report BLOCKED, choose a different route (not tiny coordinate tweaks).

COORDINATES
- Use [x,y] grid from 0..1000 for both axes.
"""

def invoke_planner(state: AgentState) -> tuple[Optional[str], Optional[ArchiveHistoryArgs]]:
    context = state.get_context()
    plan_section = f"EXISTING_PLAN:\n{state.plan[:500]}" if state.plan else "No plan exists yet."

    b64 = base64.b64encode(state.screenshot).decode("ascii") if state.screenshot else None

    if state.turn == 1:
        prompt = f"{context}\n\nFIRST_TURN_PLANNING:\nAnalyze the task carefully.\nBreak it into concrete goals with clear success criteria.\nThen use update_execution_plan to give initial instructions to the Executor."
    else:
        prompt = f"{context}\n\nPLANNING_REVIEW:\n1. Look at recent actions and their OUTCOME status\n2. Check screenshot to verify outcomes match reality\n3. If you see 3+ NO_CHANGE or any BLOCKED, provide completely different instructions\n4. If active history count is {CFG.memory_compression_threshold}+, use archive_completed_actions\n5. If you see PROGRESS pattern, continue with next goal"

    messages = [
        {"role": "system", "content": PLANNER_PROMPT.format(
            task=state.task,
            plan_section=plan_section,
            interval=CFG.review_interval,
            threshold=CFG.memory_compression_threshold
        )}
    ]

    if b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": CFG.lmstudio_model,
        "messages": messages,
        "tools": PLANNER_TOOLS,
        "tool_choice": "required",
        "temperature": 0.35,
        "max_tokens": CFG.planner_max_tokens
    }
    # resp = post_json(payload=payload)

    # msg = resp["choices"][0]["message"]
    # plan_update = msg.get("content", "").strip()
    # if plan_update:
    #     state.plan = plan_update

    # tool_calls = msg.get("tool_calls")
    # if not tool_calls:
    #     return (None, None)

    resp = post_json(payload=payload)

    choice0 = (resp.get("choices") or [{}])[0]
    finish = (choice0.get("finish_reason") or "").strip()
    msg = choice0.get("message") or {}

    plan_update = (msg.get("content") or "").strip()
    if plan_update:
        state.plan = plan_update

    tool_calls = msg.get("tool_calls")

    # Retry once if truncated or tool call missing
    if finish == "length" or not tool_calls:
        LOGGER.log(f"Planner response incomplete (finish_reason={finish}, tool_calls={bool(tool_calls)}). Retrying once...")
        payload["max_tokens"] = int(payload.get("max_tokens") or 0) + 1000
        payload["messages"].append({
            "role": "user",
            "content": (
                "Your previous output was cut off or missing a tool call. "
                "Call EXACTLY ONE tool now (update_execution_plan or archive_completed_actions). "
                "Keep it concise."
            )
        })
        resp = post_json(payload=payload)
        choice0 = (resp.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        tool_calls = msg.get("tool_calls")

    if not tool_calls:
        return (None, None)



    instructions = None
    archive_args = None

    for tc_dict in tool_calls:
        try:
            tc = ToolCall.parse(tc_dict)
            if tc.name == "update_execution_plan":
                instructions = tc.arguments.instructions
                state.plan = tc.arguments.instructions
                LOGGER.log(f"Plan updated: {tc.arguments.reasoning}")
            elif tc.name == "archive_completed_actions":
                archive_args = tc.arguments
        except Exception as e:
            LOGGER.log(f"Tool parse error: {e}")

    return (instructions, archive_args)

def invoke_executor(state: AgentState) -> tuple[Optional[ToolCall], str]:
    if not state.execution_instructions:
        return (None, "")

    context = state.get_context()
    b64 = base64.b64encode(state.screenshot).decode("ascii")

    last_action = state.memory.get_last_action_for_validation()
    previous_action_info = "No previous action (first execution turn)"
    if last_action:
        args_summary = ""
        if isinstance(last_action.args, ClickArgs):
            args_summary = f"clicked '{last_action.args.element_name}' at [{last_action.args.position.x:.0f},{last_action.args.position.y:.0f}]"
        elif isinstance(last_action.args, PressKeyArgs):
            args_summary = f"pressed key '{last_action.args.key}'"
        elif isinstance(last_action.args, TypeTextArgs):
            args_summary = f"typed '{last_action.args.text[:30]}'"
        elif isinstance(last_action.args, DragArgs):
            args_summary = f"dragged from [{last_action.args.start.x:.0f},{last_action.args.start.y:.0f}] to [{last_action.args.end.x:.0f},{last_action.args.end.y:.0f}]"
        elif isinstance(last_action.args, ScrollArgs):
            args_summary = "scrolled"
        else:
            args_summary = "performed action"

        pend = " (awaiting validation)" if (last_action.outcome or "").strip().upper() == "PENDING" else ""
        expected_change = ""
        markers_block = ""
        visual_just = ""
        try:
            if hasattr(last_action.args, "expected_visible_change"):
                expected_change = (last_action.args.expected_visible_change or "").strip()
            if hasattr(last_action.args, "change_markers"):
                ms = getattr(last_action.args, "change_markers") or ()
                markers_block = "\n".join("  - " + str(m) for m in ms) if ms else "  - <none>"
            if hasattr(last_action.args, "visual_justification"):
                # visual_just = (last_action.args.visual_justification or "").strip()
                visual_just = (last_action.args.visual_justification or "").strip()
                if len(visual_just) > 900:
                    visual_just = visual_just[:900] + " ...<truncated>"

        except Exception:
            pass

        previous_action_info = (
            f"Previous_tool: {last_action.tool}\n"
            f"Previous_action: {args_summary}\n"
            f"Previous_outcome: {last_action.outcome}{pend}\n"
            f"Previous_expected_change: {expected_change or '<unspecified>'}\n"
            f"Previous_change_markers:\n{markers_block or '  - <none>'}\n"
            f"Previous_visual_justification:\n{visual_just or '<unspecified>'}\n"
            f"Tool_result: {last_action.result}"
        )

    prompt = f"{context}\n\nCURRENT_SCREENSHOT: [shown below]\n\nYOUR_TASK_NOW:\nAssess if the previous action worked by comparing what you see now to what was expected.\nThen choose exactly ONE tool that advances toward your goal.\nSet outcome_status based on your assessment of the PREVIOUS action."

    payload = {
        "model": CFG.lmstudio_model,
        "messages": [
            {"role": "system", "content": EXECUTOR_PROMPT.format(
                task=state.task,
                instructions=state.execution_instructions,
                previous_action=previous_action_info
            )},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}
        ],
        "tools": EXECUTOR_TOOLS,
        "tool_choice": "required",
        "temperature": CFG.lmstudio_temperature,
        "max_tokens": CFG.executor_max_tokens
    }
    # resp = post_json(payload=payload)

    # msg = resp["choices"][0]["message"]
    # response_text = (msg.get("content") or "").strip()
    # tool_calls = msg.get("tool_calls")

    # if not tool_calls:
    #     return (None, response_text)
    resp = post_json(payload=payload)

    choice0 = (resp.get("choices") or [{}])[0]
    finish = (choice0.get("finish_reason") or "").strip()
    msg = choice0.get("message") or {}
    response_text = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls")

    # Retry once if truncated or tool call missing
    if finish == "length" or not tool_calls:
        LOGGER.log(f"Executor response incomplete (finish_reason={finish}, tool_calls={bool(tool_calls)}). Retrying once...")
        payload["max_tokens"] = int(payload.get("max_tokens") or 0) + 1000
        payload["messages"].append({
            "role": "user",
            "content": (
                "Your previous output was cut off or missing a tool call. "
                "Call EXACTLY ONE tool now. Keep fields concise but specific. "
                "Include: outcome_status, reasoning (PrevCheck/NowAction/Expected), visual_justification, "
                "expected_visible_change, and 3-7 change_markers."
            )
        })
        resp = post_json(payload=payload)

        choice0 = (resp.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        response_text = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls")

    if not tool_calls:
        return (None, response_text)

    try:
        return (ToolCall.parse(tool_calls[0]), response_text)
    except Exception as e:
        LOGGER.log(f"Tool parse error: {e}")
        return (None, response_text)

def execute_tool(tool_call: ToolCall, sw: int, sh: int) -> str:
    args = tool_call.arguments

    if tool_call.name in ("click_screen_element", "double_click_screen_element", "right_click_screen_element"):
        if not isinstance(args, ClickArgs) or not args.element_name:
            return "Error: element_name required"
        px, py = args.position.to_pixels(sw, sh)
        mouse_move(px, py)

        if tool_call.name == "click_screen_element":
            mouse_click("left")
            return f"Clicked {args.element_name} @[{args.position.x:.0f},{args.position.y:.0f}]"
        elif tool_call.name == "double_click_screen_element":
            mouse_double_click()
            return f"Double-clicked {args.element_name} @[{args.position.x:.0f},{args.position.y:.0f}]"
        else:
            mouse_click("right")
            return f"Right-clicked {args.element_name} @[{args.position.x:.0f},{args.position.y:.0f}]"

    elif tool_call.name == "drag_screen_element":
        if not isinstance(args, DragArgs) or not args.element_name:
            return "Error: element_name required"
        sx, sy = args.start.to_pixels(sw, sh)
        ex, ey = args.end.to_pixels(sw, sh)
        mouse_drag(sx, sy, ex, ey)
        return f"Dragged {args.element_name}"

    elif tool_call.name == "type_text_input":
        if not isinstance(args, TypeTextArgs) or not args.text:
            return "Error: text required"
        keyboard_type_text(args.text)
        return f"Typed: {args.text[:50]}"

    elif tool_call.name == "press_keyboard_key":
        if not isinstance(args, PressKeyArgs) or not args.key:
            return "Error: key required"
        key = args.key.strip().lower()
        parts = tuple(p.strip() for p in key.split("+"))
        for part in parts:
            if part not in VK_MAP:
                return f"Error: Unknown key '{part}'"
        keyboard_press_keys(key)
        return f"Pressed: {key}"

    elif tool_call.name in ("scroll_screen_down", "scroll_screen_up"):
        mouse_move(sw // 2, sh // 2)
        mouse_scroll(-1 if tool_call.name == "scroll_screen_down" else 1)
        return "Scrolled down" if tool_call.name == "scroll_screen_down" else "Scrolled up"

    elif tool_call.name == "report_goal_status":
        if not isinstance(args, ReportProgressArgs):
            return "Error: goal_identifier/completion_status/evidence required"
        return f"Goal: {args.goal_identifier} -> {args.completion_status}"

    return f"Error: unknown tool '{tool_call.name}'"

def loop_recovery_action(state: AgentState) -> None:
    if not CFG.enable_loop_recovery or not state.memory.should_recover(state.turn):
        return
    sw, sh = get_screen_size()
    try:
        keyboard_press_keys("esc")
        mouse_move(sw // 2, sh // 2)
        mouse_move(20, max(sh - 20, 0))
        mouse_click("left")
        keyboard_press_keys("ctrl+esc")
        state.memory.mark_recovery(state.turn)
        LOGGER.log("Loop recovery executed")
    except Exception as e:
        LOGGER.log(f"Loop recovery failed: {e}")

def run_agent(state: AgentState) -> str:
    for iteration in range(CFG.max_steps):
        state.increment_turn()

        if state.mode == "PLANNING":
            LOGGER.log_section(f"TURN {state.turn} - PLANNING")

            instructions, archive_args = invoke_planner(state)

            if archive_args:
                state.memory.apply_compression(
                    archive_args.summary,
                    archive_args.patterns_detected,
                    archive_args.archived_turns
                )

            if instructions:
                state.execution_instructions = instructions
                state.mode = "EXECUTION"
                LOGGER.log_section("MODE SWITCH: EXECUTION")
                LOGGER.log(f"Instructions updated:\n{instructions[:300]}")

            time.sleep(CFG.turn_delay)
            continue

        png, sw, sh = capture_png(CFG.screen_capture_w, CFG.screen_capture_h)
        screenshot_path = LOGGER.save_screenshot(png, state.turn)
        state.update_screenshot(png)
        state.screen_dims = (sw, sh)

        LOGGER.log_section(f"TURN {state.turn} - EXECUTION")
        LOGGER.log_state_update({
            "turn": state.turn,
            "active_history_size": len(state.memory.active_history)
        })

        if state.turn % CFG.review_interval == 0 or state.needs_review:
            state.mode = "PLANNING"
            state.needs_review = False
            LOGGER.log("Switching to PLANNING for review")
            continue

        tool_call, response_text = invoke_executor(state)

        if not tool_call:
            state.needs_review = True
            LOGGER.log(f"No tool call on turn {state.turn}, triggering review")
            time.sleep(CFG.turn_delay)
            continue

        # 1) Validate PREVIOUS action outcome using outcome_status provided inside THIS tool call (if present).
        prev_outcome_reported: Optional[str] = None
        prev_reasoning_note: str = ""
        if hasattr(tool_call.arguments, "outcome_status"):
            try:
                prev_outcome_reported = (tool_call.arguments.outcome_status or "").strip().upper()
            except Exception:
                prev_outcome_reported = None
        if hasattr(tool_call.arguments, "reasoning"):
            try:
                prev_reasoning_note = (tool_call.arguments.reasoning or "").strip()
            except Exception:
                prev_reasoning_note = ""

        validated = None
        if prev_outcome_reported:
            validated = state.memory.validate_last_pending(
                current_turn=state.turn,
                outcome=prev_outcome_reported,
                validation_note=prev_reasoning_note
            )
            if validated:
                LOGGER.log(f"Validated Turn {validated.turn} outcome => {validated.outcome}")

        # 2) Decide if we should schedule a planning review based on validated outcomes (less aggressive).
        # Only react to stuck signals when we actually validated a previous pending action.
        if prev_outcome_reported in ("BLOCKED", "NO_CHANGE") and validated:
            recent = state.memory.recent_outcomes(2)
            should_review = (prev_outcome_reported == "BLOCKED") or (recent.count("NO_CHANGE") >= 2)
            if should_review:
                state.needs_review = True
                LOGGER.log(f"Stuck signal detected (recent={recent}) - review scheduled")
                loop_recovery_action(state)

        if tool_call.name == "report_mission_complete":
            completion_args = tool_call.arguments
            if len(completion_args.evidence.strip()) < 100:
                LOGGER.log("Insufficient completion evidence")
                time.sleep(CFG.turn_delay)
                continue

            LOGGER.log_section("MISSION COMPLETE")
            LOGGER.log(f"Evidence: {completion_args.evidence}")
            return f"Mission completed in {state.turn} turns"

        result = execute_tool(tool_call, sw, sh)

        # 3) The action executed THIS turn has outcome PENDING until next turn validates it.
        LOGGER.log_tool_execution(
            state.turn,
            tool_call.name,
            tool_call.arguments,
            action_outcome="PENDING",
            result=result,
            prev_outcome=prev_outcome_reported
        )

        if tool_call.name == "report_goal_status" and isinstance(tool_call.arguments, ReportProgressArgs):
            status = tool_call.arguments.completion_status.strip().upper()
            if status in ("DONE", "COMPLETED", "COMPLETE", "BLOCKED"):
                state.needs_review = True
                LOGGER.log(f"Goal status {status} - review scheduled")

        state.memory.add_action(ActionRecord(
            turn=state.turn,
            tool=tool_call.name,
            args=tool_call.arguments,
            outcome="PENDING",
            result=result,
            screenshot=screenshot_path
        ))

        # Tiny patch: coordinate repetition detector to catch click-loops early
        rep = state.memory.detect_repeated_clicks(n=3, tol=10.0)
        if rep:
            state.needs_review = True
            LOGGER.log(rep + " - review scheduled")
            loop_recovery_action(state)

        time.sleep(CFG.turn_delay)

    return f"Max steps reached ({CFG.max_steps})"

def main() -> None:
    init_dpi()

    LOGGER.log_section("INITIALIZATION")
    LOGGER.log(f"Max Steps: {CFG.max_steps}")
    LOGGER.log(f"Review Interval: {CFG.review_interval}")
    LOGGER.log(f"Planner Tokens: {CFG.planner_max_tokens}")
    LOGGER.log(f"Executor Tokens: {CFG.executor_max_tokens}")

    task = input("Task: ").strip()
    if not task:
        sys.exit("Error: Task description required")

    LOGGER.log(f"Task: {task}")

    png, sw, sh = capture_png(CFG.screen_capture_w, CFG.screen_capture_h)
    LOGGER.save_screenshot(png, 0)

    state = AgentState(task=task, screenshot=png, screen_dims=(sw, sh), mode="PLANNING")

    LOGGER.log_section("OPERATIONS START")
    result = run_agent(state)

    LOGGER.log_section("DEBRIEF")
    LOGGER.log(f"Status: {result}")
    LOGGER.log(f"Turns: {state.turn}")
    LOGGER.log(f"Archives: {len(state.memory.all_snapshots)}")

if __name__ == "__main__":
    main()
