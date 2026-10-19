from __future__ import annotations

import re
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional, Sequence

import requests

from context import create_context
from core.prompting.composer import model_message, token_retry_prompt
from core.prompting.content import Prompt, message_blocks

# Rate-limit / transient-error retry defaults. The
# client retries 429 and 5xx with exponential backoff, honouring a Retry-After header or
# a provider "retry in Xs" / "retryDelay" hint when present.
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_S = 2.0
DEFAULT_RETRY_MAX_S = 30.0
# A provider-specified wait (Retry-After / "retry in Xs") is honoured up to this ceiling.
# It must exceed a typical per-minute RPM window (~60 s) -- clamping a hinted wait below
# the stated value would just guarantee the retry is still rate-limited. The ceiling only
# guards against a pathological hint (e.g. a daily-quota "retry in 3600s").
RETRY_HINT_CEILING_S = 90.0
# Besides 5xx: request timeout, transient conflict, too-early, and rate limit are all
# worth retrying; every other 4xx is a caller/auth error that a retry cannot fix.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})


@dataclass
class VLMResponse:
    token: str
    raw_text: str
    payload: dict


class VLMParseError(RuntimeError):
    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


class VLMClient:
    """HTTP transport and response parsing for the configured hosted model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_s: float,
        max_tokens: int,
        temperature: float,
        max_retries: Optional[int] = None,
        retry_base_delay_s: Optional[float] = None,
        retry_max_delay_s: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max(0, int(max_retries))
        self.retry_base_delay_s = (
            DEFAULT_RETRY_BASE_S if retry_base_delay_s is None else float(retry_base_delay_s)
        )
        self.retry_max_delay_s = (
            DEFAULT_RETRY_MAX_S if retry_max_delay_s is None else float(retry_max_delay_s)
        )
        self._api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def _finalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply the configured model's request dialect in LiberoVLMClient."""
        raise NotImplementedError

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /chat/completions and return the parsed completion body, with
        retry+backoff on rate limits (HTTP 429), transient HTTP errors (408/409/425/5xx),
        network errors, and malformed "successful" bodies -- a relaying proxy (e.g. the
        hosted API proxy) can answer 2xx with an error JSON, an HTML page, or an
        empty choices list while its upstream is rate-limited or restarting, so those are
        retried too instead of crashing the rollout in the caller's parse. Honours a
        Retry-After header or a provider 'retry in Xs' / 'retryDelay' body hint; otherwise
        exponential backoff with jitter. Raises RuntimeError on a non-retryable error
        (e.g. 400/401/404) or once ``max_retries`` is exhausted."""
        url = f"{self.base_url}/chat/completions"
        delay = self.retry_base_delay_s
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            # Default wait is exponential backoff, capped at retry_max_delay_s.
            wait = min(delay, self.retry_max_delay_s)
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout_s)
            except requests.ConnectionError as exc:
                last = f"request error: {exc}"  # network/tunnel hiccup -> retry
                if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                    # A refused localhost connection is a DOWN TUNNEL/PROXY, not a
                    # rate limit -- without this hint the retry banner reads like one.
                    last = (
                        f"nothing listening at {self.base_url} -- the local "
                        "server or tunnel/proxy for this backend is DOWN; "
                        "start it and retry"
                    )
            except requests.RequestException as exc:
                last = f"request error: {exc}"  # network/tunnel hiccup -> retry
            else:
                if resp.status_code < 400:
                    data, problem = _chat_completion_data(resp)
                    if data is not None:
                        return data
                    last = f"bad completion body: {problem}"  # proxy glitch -> retry
                else:
                    last = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                    if not (
                        resp.status_code in RETRYABLE_STATUS_CODES or 500 <= resp.status_code < 600
                    ):
                        raise RuntimeError(f"VLM chat completion failed: {last}")
                    hinted = _retry_after_seconds(resp)
                    if hinted is not None:
                        # Honour the provider's stated wait FULLY (+1s margin) so the
                        # retry lands after the rate window resets -- but never retry
                        # SOONER than the exponential schedule: a congested shared pool
                        # keeps answering "retry after 1 seconds" while
                        # staying exhausted for a whole per-minute window, and a
                        # hint-pinned ~2s cadence burns every retry inside it.
                        wait = min(max(hinted + 1.0, wait), RETRY_HINT_CEILING_S)
            if attempt >= self.max_retries:
                break
            # Jitter de-synchronises concurrent clients (e.g. benchmark workers sharing one
            # proxy) so they do not all land in the same rate window again.
            wait = max(wait, 0.0) * random.uniform(0.8, 1.2)
            # Collapse whitespace so a pretty-printed JSON error body
            # shows its message instead of just its opening "{" line.
            print(
                f"[vlm] rate-limited/transient ({' '.join(last.split())[:160]}); "
                f"retry {attempt + 1}/{self.max_retries} in {wait:.1f}s"
            )
            time.sleep(wait)
            delay = min(delay * 2, self.retry_max_delay_s)
        raise RuntimeError(f"VLM chat completion failed after {self.max_retries} retries: {last}")

    def complete_token(
        self,
        prompt: Prompt,
        allowed_tokens: Sequence[str],
        agentview_image,
        wrist_image=None,
        debug: bool = False,
        context: dict | None = None,
    ) -> VLMResponse:
        context = create_context() if context is None else context
        if not allowed_tokens:
            raise ValueError("allowed_tokens must not be empty")
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            context=context,
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if debug:
            payload["logprobs"] = True
        data, raw_text, latency_s = self._post_completion(payload)
        try:
            token = _parse_single_token(raw_text, allowed_tokens)
        except RuntimeError as exc:
            retry_prompt = token_retry_prompt({
                **(context or {}), "previous_prompt": prompt, "allowed_tokens": allowed_tokens,
            })
            retry_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": _message_content(
                            retry_prompt,
                            agentview_image,
                            wrist_image,
                            context=context,
                        ),
                    }
                ],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }
            if debug:
                retry_payload["logprobs"] = True
            retry_data, retry_raw_text, retry_latency_s = self._post_completion(retry_payload)
            try:
                token = _parse_single_token(retry_raw_text, allowed_tokens)
            except RuntimeError as retry_exc:
                raise RuntimeError(
                    f"{exc}; strict token retry also failed: {retry_exc}"
                ) from retry_exc
            raw_text = f"{retry_raw_text.strip()} [strict_token_retry]"
            data = {"first": data, "retry": retry_data} if debug else {}
            latency_s += retry_latency_s
        return VLMResponse(
            token=token,
            raw_text=raw_text,
            payload=_response_payload(data if debug else {}, latency_s),
        )

    def complete_text(
        self,
        prompt: Prompt,
        agentview_image,
        wrist_image=None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        debug: bool = False,
        strip_reasoning: bool = True,
        image_detail: Optional[str] = None,
        context: dict | None = None,
    ) -> VLMResponse:
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            image_detail=image_detail,
            context=context,
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": (self.temperature if temperature is None else float(temperature)),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        if debug:
            payload["logprobs"] = True
        payload = self._finalize_payload(payload)
        started = time.monotonic()
        data = self._post_chat(payload)
        latency_s = time.monotonic() - started
        # strip_reasoning=False keeps the <think>...</think> chain-of-thought in the
        # text (for CoT logging/analysis); token recovery still finds the answer.
        message_text = _message_text(data["choices"][0]["message"])
        raw_text = _strip_reasoning_artifacts(message_text) if strip_reasoning else message_text
        return VLMResponse(
            token="",
            raw_text=raw_text.strip(),
            payload=_response_payload(data if debug else {}, latency_s),
        )

    def complete_json(
        self,
        prompt: Prompt,
        agentview_image,
        wrist_image=None,
        schema: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        debug: bool = False,
        image_detail: Optional[str] = None,
        context: dict | None = None,
    ) -> VLMResponse:
        content = _message_content(
            prompt,
            agentview_image,
            wrist_image,
            image_detail=image_detail,
            context=context,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": (self.temperature if temperature is None else float(temperature)),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        if debug:
            payload["logprobs"] = True
        payload = self._finalize_payload(payload)
        started = time.monotonic()
        data = self._post_chat(payload)
        latency_s = time.monotonic() - started
        raw_text = _strip_reasoning_artifacts(_message_text(data["choices"][0]["message"])).strip()
        parsed = _parse_json_object(raw_text)
        clean_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        response_payload = {"json": parsed, "raw": data} if debug else {"json": parsed}
        return VLMResponse(
            token="",
            raw_text=clean_text,
            payload=_response_payload(response_payload, latency_s),
        )

    def _post_completion(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
        payload = self._finalize_payload(payload)
        started = time.monotonic()
        data = self._post_chat(payload)
        latency_s = time.monotonic() - started
        raw_text = _strip_reasoning_artifacts(_message_text(data["choices"][0]["message"]))
        return data, raw_text, latency_s


def _chat_completion_data(response) -> tuple[Optional[dict[str, Any]], str]:
    """Parse and validate a 2xx /chat/completions body.

    Returns ``(data, "")`` when the body is a usable completion, else ``(None,
    problem)`` so ``_post_chat`` retries it as transient: a relaying proxy can answer
    2xx with an error JSON, an HTML error page (non-JSON), or a choices-less body
    while its upstream is rate-limited or restarting, and indexing
    ``choices[0]["message"]`` on those used to crash the rollout."""
    try:
        data = response.json()
    except ValueError as exc:
        return None, f"non-JSON body: {exc}"
    if not isinstance(data, dict):
        return None, f"unexpected body type: {type(data).__name__}"
    if data.get("error"):
        return None, f"error payload: {json.dumps(data['error'], ensure_ascii=False)[:300]}"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, "response has no choices"
    if not isinstance(choices[0].get("message"), dict):
        return None, "first choice has no message"
    return data, ""


def _retry_after_seconds(response) -> Optional[float]:
    """Best-effort: how long to wait before retrying, from the ``Retry-After`` header
    (delta-seconds or HTTP-date form) or a provider body hint ( ``retry in
    23.2s`` / ``"retryDelay": "23s"``). Returns ``None`` when nothing parseable so the
    caller falls back to exponential backoff."""
    header = (response.headers.get("Retry-After") or "").strip()
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            try:
                when = parsedate_to_datetime(header)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    try:
        body = response.text or ""
    except Exception:  # noqa: BLE001 - body may be unavailable
        body = ""
    match = (
        re.search(r"retry in ([0-9.]+)\s*s", body, re.IGNORECASE)
        # Alternate phrasing: 'Rate Limit Exceeded, retry after 1 seconds.'
        or re.search(r"retry after ([0-9.]+)\s*second", body, re.IGNORECASE)
        or re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body)
    )
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _response_payload(payload: dict[str, Any], latency_s: float) -> dict[str, Any]:
    result = dict(payload)
    result["latency_s"] = round(float(latency_s), 3)
    return result


def _message_content(
    prompt: Prompt,
    agentview_image,
    wrist_image=None,
    *,
    image_detail: Optional[str] = None,
    context: dict | None = None,
) -> list[dict[str, Any]]:
    context = create_context() if context is None else context
    content = model_message({
        **(context or {}), "previous_prompt": prompt, "agentview_image": agentview_image,
        "wrist_image": wrist_image, "image_detail": image_detail,
    })
    return message_blocks(content)


def _parse_single_token(raw_text: str, allowed_tokens: Sequence[str]) -> str:
    stripped = _strip_reasoning_artifacts(raw_text).strip()
    if stripped in allowed_tokens:
        return stripped
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("token", "answer", "direction", "status", "decision"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip() in allowed_tokens:
                return value.strip()
    pattern = r"^(?:" + "|".join(re.escape(token) for token in allowed_tokens) + r")$"
    if re.match(pattern, stripped):
        return stripped
    suffix_pattern = (
        r"(?:^|[^A-Z_])("
        + "|".join(re.escape(token) for token in allowed_tokens)
        + r")\s*[.。]*\s*$"
    )
    suffix_match = re.search(suffix_pattern, stripped)
    if suffix_match:
        return suffix_match.group(1)
    recovered = recover_allowed_token(stripped, allowed_tokens)
    if recovered:
        return recovered
    raise RuntimeError(
        f"VLM returned invalid token {raw_text!r}; allowed tokens are {list(allowed_tokens)}"
    )


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    original_text = raw_text
    raw_text = _strip_reasoning_artifacts(raw_text).strip()
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    if value is not None:
        raise RuntimeError(f"VLM returned JSON that is not an object: {original_text!r}")

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", raw_text):
        try:
            candidate, _ = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if "subgoals" in candidate:
            return candidate
        candidates.append(candidate)
    if candidates:
        return candidates[0]
    raise VLMParseError(
        f"VLM returned non-JSON text: {original_text!r}",
        raw_text=original_text,
    )


def recover_allowed_token(raw_text: str, allowed_tokens: Sequence[str]) -> str:
    tokens = tuple(allowed_tokens)
    if not tokens:
        return ""
    token_pattern = "|".join(re.escape(token) for token in tokens)
    explicit = re.search(
        rf'"(?:token|answer|direction|status|decision)"\s*:\s*"?\s*({token_pattern})\b',
        raw_text,
    )
    if explicit:
        return explicit.group(1)
    choice_patterns = (
        rf"\b(?:therefore|thus|so|final(?:ly)?|choose|chosen|select|selected|use|"
        rf"decision|answer)\b[^.\n]*?\b({token_pattern})\b",
        rf"\b({token_pattern})\b[^.\n]*?\b(?:best|correct|required|appropriate)\b",
    )
    for pattern in choice_patterns:
        matches = list(re.finditer(pattern, raw_text, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1)
    sentences = [part.strip() for part in re.split(r"[.\n]+", raw_text) if part.strip()]
    if sentences:
        final_tokens = _tokens_in_text(sentences[-1], tokens)
        if len(final_tokens) == 1:
            return final_tokens[0]
    found = _tokens_in_text(raw_text, tokens)
    unique = list(dict.fromkeys(found))
    if len(unique) == 1:
        return unique[0]
    return ""


def _tokens_in_text(raw_text: str, allowed_tokens: Sequence[str]) -> list[str]:
    result: list[str] = []
    for token in allowed_tokens:
        if re.search(rf"\b{re.escape(token)}\b", raw_text):
            result.append(token)
    return result


def _message_text(message: dict) -> str:
    """Return the model's answer text, tolerating reasoning (CoT) models.

    A reasoner served with a reasoning parser splits chain-of-thought into
    ``reasoning_content`` and the answer into ``content``. If ``content`` comes
    back empty (guided decoding, or the server routing everything into the
    reasoning channel), fall back to ``reasoning_content`` so the JSON/token
    parsers downstream still receive the output instead of an empty string.
    """
    content = message.get("content")
    if content and str(content).strip():
        return content
    reasoning = message.get("reasoning_content")
    return reasoning or content or ""


def _strip_reasoning_artifacts(raw_text: str) -> str:
    text = str(raw_text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
