"""Decode semantic Responses SSE events without assembling partial model output."""

from collections.abc import Iterable, Iterator

from harness.serialization import Object, parse_object


def response_events(lines: Iterable[str]) -> Iterator[Object]:
    name = ""
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                event = parse_object("\n".join(data))
                event.setdefault("type", name)
                yield event
            name, data = "", []
        elif line.startswith("event:"):
            name = line[6:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        # Ignore SSE comments, ids, and retry hints.
    if data:
        raise RuntimeError("Provider stream ended inside an event")
