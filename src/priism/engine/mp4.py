"""Minimal MP4 box editing: add the NI ``stem`` box under ``moov/udta``.

Only ``moov`` (and the ``udta`` inside it) grow, so the file must have ``moov`` after
``mdat`` (ffmpeg's default without ``+faststart``): the sample offsets in ``stco``/``co64``
point into ``mdat`` and stay valid. Mixxx's ``StemInfoImporter`` reads the first
``moov/udta/stem`` it meets, so the box goes into an existing ``udta`` rather than a second one.
"""

from __future__ import annotations

import struct


class Mp4Error(ValueError):
    pass


def _boxes(data: bytes, start: int, end: int) -> list[tuple[str, int, int, int]]:
    """(type, box start, payload start, box end) of the boxes laid out in data[start:end]."""
    out = []
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise Mp4Error(f"corrupt box {kind!r} at offset {pos}")
        out.append((kind.decode("latin-1"), pos, pos + header, pos + size))
        pos += size
    return out


def _box(kind: str, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind.encode("latin-1")) + payload


def _resize(data: bytearray, box_start: int, delta: int) -> None:
    size = struct.unpack(">I", data[box_start:box_start + 4])[0]
    if size in (0, 1):
        raise Mp4Error("cannot resize a box with a 64-bit or open-ended size")
    data[box_start:box_start + 4] = struct.pack(">I", size + delta)


def add_stem_box(data: bytes, payload: bytes) -> bytes:
    top = _boxes(data, 0, len(data))
    kinds = [k for k, *_ in top]
    if "moov" not in kinds or "mdat" not in kinds:
        raise Mp4Error("not an MP4 file with moov and mdat")
    if kinds.index("moov") < kinds.index("mdat"):
        raise Mp4Error("moov before mdat (faststart): growing it would break the sample offsets")
    _, moov_start, moov_payload, moov_end = top[kinds.index("moov")]
    children = _boxes(data, moov_payload, moov_end)
    out = bytearray(data)
    stem = _box("stem", payload)
    udta = [c for c in children if c[0] == "udta"]
    if udta:
        _, u_start, u_payload, u_end = udta[0]
        if any(k == "stem" for k, *_ in _boxes(data, u_payload, u_end)):
            raise Mp4Error("the file already has a stem box")
        out[u_end:u_end] = stem
        _resize(out, u_start, len(stem))
        _resize(out, moov_start, len(stem))
    else:
        box = _box("udta", stem)
        out[moov_end:moov_end] = box
        _resize(out, moov_start, len(box))
    return bytes(out)


def read_stem_box(data: bytes) -> bytes | None:
    """Payload of moov/udta/stem, the way Mixxx looks it up."""
    for kind, _, p, e in _boxes(data, 0, len(data)):
        if kind == "moov":
            for k2, _, p2, e2 in _boxes(data, p, e):
                if k2 == "udta":
                    for k3, _, p3, e3 in _boxes(data, p2, e2):
                        if k3 == "stem":
                            return data[p3:e3]
    return None
