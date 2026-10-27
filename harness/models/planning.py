"""Run DeepSeek Harness with a read-only profile and the shared strict transport."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import shutil
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from harness.context import PluginContext
from harness.models.deepseek import CompletionClient
from harness.prompting import observation_blocks, task_input, wire_blocks
from harness.robot_prompt import describe_actions, describe_robot
from harness.schemas import validate
from harness.serialization import Object, dumps, parse_object
from harness.trajectory import TrajectoryStore


class HarnessPlanningClient:
    def __init__(self, client: CompletionClient, trajectory: TrajectoryStore, executable: str = "dsh"):
        self.client, self.trajectory = client, trajectory
        self.executable = executable
        self._sequence = 0

    def plan(self, instructions: str, request: str, context: PluginContext, output_schema: Object) -> Object:
        executable = shutil.which(self.executable)
        if executable is None:
            raise FileNotFoundError(f"DeepSeek Harness executable unavailable: {self.executable}")
        self._sequence += 1
        directory = self.trajectory.directory / "helpers" / f"subgoal_{self._sequence:04d}"
        directory.mkdir()
        history = self.trajectory.history_path.read_text()
        (directory / "history.jsonl").write_text(history)
        state = context.state
        sections = [f"Planning request: {request}", describe_robot(state.observation.robot)]
        if context.previous_result:
            sections.append(
                f"The previous subgoal was {context.previous_result['subgoal']}\n"
                f"Its completion was defined as {context.previous_result['completion']}"
            )
        if actions := describe_actions(state.recent_actions):
            sections.append(actions)
        question = "\n\n".join(sections)
        prefix: list[Object] = [
            task_input(state.task.instruction),
            {"role": "user", "content": wire_blocks(observation_blocks(state.observation))},
        ]
        secret = secrets.token_urlsafe(32)
        client = self.client

        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                response: Object
                if self.path != "/complete" or not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + secret
                ):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError("Invalid gateway request size")
                    body = parse_object(self.rfile.read(length).decode())
                    reply = client.complete(
                        role="subgoal",
                        instructions=instructions,
                        input_items=[*prefix, *body["input"]],
                        tools=body["tools"],
                        output_schema=output_schema,
                        decision_id=state.decision_id,
                    )
                    response = {"response": reply.raw}
                    status = 200
                except Exception as error:
                    response = {"error": str(error)}
                    status = 500
                encoded = dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        server.daemon_threads = False
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        runtime_home = directory / "dsh_home"
        profile = runtime_home / "profiles" / "robot-planner"
        profile.mkdir(parents=True)
        (profile / "package.json").write_text(
            dumps(
                {
                    "name": "seed-robot-planner",
                    "private": True,
                    "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-sdk-minimal"]}},
                }
            )
        )
        disabled = [
            "sdk-app-startup",
            "sdk-jsonrpc-server",
            "llm-deepseek",
            "deepseek-llm-api-extensions",
            "session-log-deepseek",
            "plugin-package-inventory-deepseek",
            "persistent-bash",
            "persistent-pwsh",
            "terminal-bash",
            "terminal-pwsh",
            "pty",
            "mcp-resources",
        ]
        patch: list[Object] = [{"id": name, "disabled": True} for name in disabled]
        patch.extend(
            [
                {
                    "id": "system-prompt",
                    "config": {
                        "includeHarnessIdentity": False,
                        "includeRuntimeContext": False,
                        "personaPrefix": instructions,
                    },
                },
                {"id": "tools", "config": {"mode": "native"}},
                {
                    "insert": [
                        {
                            "id": "agent-default-model",
                            "name": "@deepseek-ai/dsh-agent-default-model",
                            "config": {"provider": "seed-strict", "model": "deepseek-flash"},
                        },
                        {
                            "id": "seed-planner",
                            "name": str(Path(__file__).with_name("planner_bridge.mjs")),
                            "config": {
                                "root": str(self.trajectory.directory),
                                "history": str(directory / "history.jsonl"),
                                "gateway": f"http://127.0.0.1:{server.server_port}/complete",
                            },
                        },
                        {"id": "headless-startup", "name": "@deepseek-ai/dsh-headless/startup"},
                        {
                            "id": "headless-runner",
                            "name": "@deepseek-ai/dsh-headless",
                            "inject": ["headlessStartup"],
                            "config": {"task": question, "json": True},
                        },
                    ]
                },
            ]
        )
        (profile / "cordis.patch.yml").write_text(yaml.safe_dump(patch, sort_keys=False))
        env = dict(os.environ)
        env.update(
            DSH_HOME=str(runtime_home),
            SEED_GATEWAY_TOKEN=secret,
            DSH_PRIMARY_RUNTIME="",
            DSH_PERMISSION_MODE="read-only",
        )
        env.pop("DEEPSEEK_API_KEY", None)
        self.trajectory.record(
            "planning_agent_started",
            decision_id=state.decision_id,
            directory=str(directory.relative_to(self.trajectory.directory)),
            history_lines=len(history.splitlines()),
        )
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [executable, "--profile", "robot-planner", "--json", "-"],
                cwd=self.trajectory.directory,
                env=env,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = process.communicate(question)
            (directory / "agent_events.jsonl").write_text(stdout)
            (directory / "stderr.txt").write_text(stderr)
            if process.returncode:
                raise RuntimeError(f"DeepSeek Harness exited {process.returncode}: {stderr[-2500:]}")
            events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            finals = [event for event in events if event.get("type") == "final"]
            if not finals:
                raise ValueError("DeepSeek Harness returned no final event")
            final = finals[-1]
            text = final.get("text", final.get("answer", ""))
            result = parse_object(text)
            validate(output_schema, result)
            self.trajectory.record(
                "planning_agent_completed",
                decision_id=state.decision_id,
                result=result,
                agent_events=str((directory / "agent_events.jsonl").relative_to(self.trajectory.directory)),
            )
            return result
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
