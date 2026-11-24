#!/usr/bin/env python3
"""Inventory and copy only the fixed Inkling-2-Mini GGUF transplants."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from tools.gguf_reader import BLOCK_GEOMETRY, GGML_TYPES, GGUF_META, GGUFFile


RECORDS = Path(__file__).resolve().parents[2] / 'docs/goals/inkling2-weights'
HEADER_BYTES = 12 << 20
COPY_CHUNK = 8 << 20
ARCH = b'inkling2_loop'
META_PREFIXES = ('tokenizer.', 'inkling2_loop.')
GROUPS = {
    'H1': tuple(f'blk.{i}.attn_sink_proj.weight' for i in range(24)) + tuple(
        f'blk.{i}.attn_{kv}_conv_proj.weight' for i in (5, 11, 17, 23) for kv in ('k', 'v')),
    'H2': ('token_embd.weight',),
    'H3': tuple(f'blk.{i}.attn_{p}.weight' for i in range(6, 18) for p in ('q', 'gate', 'rel')),
}
DELTAS = {'H1': 2755584, 'H2': 231579648, 'H3': 82944000}
A1_NAMES = tuple(f'blk.{i}.attn_output.weight' for i in range(24))
DONOR_KEYS = ('repo', 'revision', 'filename', 'size_bytes', 'advertised_sha256')


def align(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


class Cursor:
    def __init__(self, raw: bytes):
        self.raw = memoryview(raw)
        self.pos = 0

    def take(self, size: int) -> memoryview:
        end = self.pos + size
        if end > len(self.raw):
            raise ValueError('truncated or oversized GGUF header')
        value = self.raw[self.pos:end]
        self.pos = end
        return value

    def read(self, fmt: str) -> tuple:
        fmt = '<' + fmt
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def string(self) -> memoryview:
        return self.take(self.read('Q')[0])

    def value(self, kind: int) -> None:
        if kind not in GGUF_META:
            raise ValueError(f'unknown GGUF metadata type {kind}')
        if kind == 8:
            self.string()
        elif kind == 9:
            element, count = self.read('IQ')
            if element not in GGUF_META or element == 9:
                raise ValueError(f'unsupported GGUF metadata array element type {element}')
            if element == 8:
                for _ in range(count):
                    self.string()
            else:
                self.take(GGUF_META[element][1] * count)
        else:
            self.take(GGUF_META[kind][1])


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int  # relative to data_start

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, f'type{self.type_id}')

    @property
    def nbytes(self) -> int:
        geometry = BLOCK_GEOMETRY.get(self.type_name)
        if geometry is None:
            raise ValueError(f'{self.name}: unsupported tensor type {self.type_id}')
        block, size = geometry
        if not all(self.shape) or self.shape[0] % block:
            raise ValueError(f'{self.name}: invalid row geometry for {self.type_name}: {self.shape}')
        return self.shape[0] // block * size * math.prod(self.shape[1:])

    def span(self, data_start: int) -> dict:
        return {'type_id': self.type_id, 'type': self.type_name, 'shape': self.shape,
                'offset': self.offset, 'absolute_offset': data_start + self.offset, 'bytes': self.nbytes}


@dataclass(frozen=True)
class Header:
    alignment: int
    data_start: int
    metadata: dict[str, memoryview]  # complete serialized type and value, not coerced Python values
    metadata_block: memoryview  # ordered typed records, including keys, but not their count
    tensors: dict[str, Tensor]
    file_size: int


def parse_header(raw: bytes, file_size: int) -> Header:
    cursor = Cursor(raw)
    magic, version, n_tensors, n_kv = cursor.read('4sIQQ')
    if magic != b'GGUF' or version != 3:
        raise ValueError('expected a little-endian GGUF v3 file')
    start = cursor.pos
    metadata = {}
    for _ in range(n_kv):
        key = bytes(cursor.string()).decode('utf-8')
        if key in metadata:
            raise ValueError(f'duplicate metadata key: {key}')
        at = cursor.pos
        cursor.value(cursor.read('I')[0])
        metadata[key] = cursor.raw[at:cursor.pos]
    metadata_block = cursor.raw[start:cursor.pos]
    alignment = 32
    if 'general.alignment' in metadata:
        kind, alignment = struct.unpack('<II', metadata['general.alignment'])
        if kind != 4 or not alignment or alignment & (alignment - 1):
            raise ValueError('general.alignment must be a power-of-two UINT32')
    tensors = {}
    for _ in range(n_tensors):
        name = bytes(cursor.string()).decode('utf-8')
        if name in tensors:
            raise ValueError(f'duplicate tensor: {name}')
        dims = cursor.read('I')[0]
        if not 1 <= dims <= 4:
            raise ValueError(f'{name}: invalid dimensionality {dims}')
        shape = cursor.read(f'{dims}Q')
        type_id, offset = cursor.read('IQ')
        tensors[name] = Tensor(name, shape, type_id, offset)
    data_start = align(cursor.pos, alignment)
    if data_start > min(len(raw), file_size):
        raise ValueError('incomplete aligned GGUF header')
    end = 0
    for tensor in sorted(tensors.values(), key=lambda t: t.offset):
        if tensor.offset % alignment or tensor.offset < end:
            raise ValueError(f'{tensor.name}: unaligned or overlapping GGUF span')
        end = tensor.offset + tensor.nbytes
        if data_start + end > file_size:
            raise ValueError(f'{tensor.name}: GGUF span exceeds source size')
    return Header(alignment, data_start, metadata, metadata_block, tensors, file_size)


@dataclass(frozen=True)
class PublicRange:
    repo: str
    revision: str
    filename: str
    size: int
    advertised_sha256: str | None = None
    token: bytes | None = field(default=None, repr=False)

    def read(self, start: int, length: int) -> bytes:
        if not (0 < length <= HEADER_BYTES and 0 <= start <= self.size - length):
            raise ValueError('invalid or oversized HTTP range')
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', self.repo) or not re.fullmatch(r'[\w.-]+', self.filename):
            raise ValueError('expected a plain HF repo and filename')
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise ValueError('expected a full pinned HF commit')
        stop = start + length - 1
        url = f'https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}'
        # curl also supports the host's standard SOCKS proxy configuration without Python dependencies.
        command = ['curl', '--disable', '--silent', '--show-error', '--fail', '--location',
                   '--proto', '=https', '--proto-redir', '=https', '--max-time', '120',
                   '--range', f'{start}-{stop}', '--max-filesize', str(length), '--suppress-connect-headers',
                   '--dump-header', '/dev/stderr']
        options = {}
        if self.token is not None:
            if not re.fullmatch(rb'hf_[a-zA-Z0-9]+', self.token):
                raise ValueError('expected a Hugging Face token on stdin (A1 only)')
            # curl --location withholds Authorization on redirects to another origin.
            command += ['--header', '@-']
            options['input'] = b'Authorization: Bearer ' + self.token + b'\n'
        got = subprocess.run([*command, url], capture_output=True, timeout=125, **options)
        if got.returncode:
            raise ValueError(f'bounded HF Range {start}-{stop} failed (curl exit {got.returncode})')
        # Never print raw response headers: redirect URLs may include signed parameters.
        responses = []
        for block in re.split(rb'\r?\n\r?\n', got.stderr):
            status, *lines = block.splitlines() or [b'']
            match = re.fullmatch(rb'HTTP/\S+ ([0-9]{3})(.*)', status)
            if not match or int(match[1]) < 200 or match[2].strip().lower() == b'connection established':
                continue
            headers = {}
            for line in lines:
                key, colon, value = line.partition(b':')
                if colon:
                    headers.setdefault(key.lower(), []).append(value.strip(b' \t'))
            responses.append((int(match[1]), headers))
        if not responses or responses[-1][0] != 206:
            raise ValueError(f'expected HTTP 206 for Range {start}-{stop}')
        if responses[0][1].get(b'x-repo-commit') != [self.revision.encode()]:
            raise ValueError('HF Range response did not confirm the pinned commit')
        if self.advertised_sha256 is not None:
            origin = responses[0][1]
            if (origin.get(b'x-linked-size') != [str(self.size).encode()] or
                    origin.get(b'x-linked-etag') != [f'"{self.advertised_sha256}"'.encode()]):
                raise ValueError('HF Range origin did not confirm the advertised size and LFS digest')
        expected = f'bytes {start}-{stop}/{self.size}'.encode()
        if responses[-1][1].get(b'content-range') != [expected] or len(got.stdout) != length:
            raise ValueError(f'wrong Content-Range or response length for Range {start}-{stop}')
        return got.stdout


def donor_range(pin: dict) -> PublicRange:
    return PublicRange(pin['repo'], pin['revision'], pin['filename'], pin['size_bytes'])


def layout(header: Header, edits: dict[str, Tensor]) -> tuple[dict[str, Tensor], int]:
    missing = edits.keys() - header.tensors.keys()
    if missing:
        raise ValueError(f'cannot replace missing tensors: {sorted(missing)}')
    placed = {}
    offset = 0
    for base in sorted(header.tensors.values(), key=lambda t: t.offset):
        tensor = edits.get(base.name, base)
        if (tensor.name, tensor.shape) != (base.name, base.shape):
            raise ValueError(f'{base.name}: replacement name or shape mismatch')
        placed[base.name] = replace(tensor, offset=offset)
        offset = align(offset + tensor.nbytes, header.alignment)
    return placed, header.data_start + offset


def inventory(base_path: Path, pins: dict, recon: dict) -> dict:
    with base_path.open('rb') as base_file:
        raw_base = base_file.read(HEADER_BYTES)
        size = base_file.seek(0, 2)
    if size != pins['H0']['size_bytes']:
        raise ValueError('H0 file size does not match its pin')
    remote = donor_range(pins['U'])
    raw_donor = remote.read(0, HEADER_BYTES)
    digests = {label: sha256(raw).hexdigest() for label, raw in (('H0', raw_base), ('U', raw_donor))}
    if digests != recon['header_sha256']:
        raise ValueError('local H0 or pinned U header hash differs from RECON')
    base = parse_header(raw_base, size)
    donor = parse_header(raw_donor, remote.size)
    arch = struct.pack('<IQ', 8, len(ARCH)) + ARCH
    if base.metadata.get('general.architecture') != arch or donor.metadata.get('general.architecture') != arch:
        raise ValueError('expected inkling2_loop in both sources')
    keys = sorted(k for k in base.metadata.keys() | donor.metadata.keys() if k.startswith(META_PREFIXES))
    incompatible = [k for k in keys if base.metadata.get(k) != donor.metadata.get(k)]
    if incompatible or sum(k.startswith('tokenizer.') for k in keys) != 12:
        raise ValueError(f'architecture/tokenizer metadata mismatch: {incompatible}')
    if len(base.tensors) != 704 or len(donor.tensors) != 744 or any(
        name not in donor.tensors or donor.tensors[name].shape != t.shape for name, t in base.tensors.items()
    ):
        raise ValueError('unexpected tensor inventory or mismatched shared shapes')
    extra = donor.tensors.keys() - base.tensors.keys()
    if extra != {f'blk.{i}.ffn_{part}_exps.scale' for i in range(20) for part in ('down', 'gate_up')}:
        raise ValueError('unexpected donor-only tensors')
    if recon['H1_payload_sha256'].keys() != set(GROUPS['H1']):
        raise ValueError('RECON H1 hashes do not cover exactly the H1 whitelist')

    edits = {}
    groups = {}
    for group, names in GROUPS.items():
        for name in names:
            old, new = base.tensors[name], donor.tensors[name]
            allowed_base = (2, 14) if group == 'H3' and name.endswith('.attn_q.weight') else (2,)
            if old.type_id not in allowed_base or new.type_id != (30 if group == 'H1' else 8):
                raise ValueError(f'{name}: unexpected precision pair {old.type_name} → {new.type_name}')
            edits[name] = new
        if group == 'H2' and edits['token_embd.weight'].shape != (2304, 201024):
            raise ValueError('embedding must include all 201024 rows')
        delta = sum(edits[n].nbytes - base.tensors[n].nbytes for n in names)
        if delta != DELTAS[group]:
            raise ValueError(f'{group}: unexpected payload delta {delta}')
        placed, end = layout(base, edits)
        tensors = []
        for name in names:
            item = {'name': name, 'H0': base.tensors[name].span(base.data_start),
                    'U': donor.tensors[name].span(donor.data_start),
                    'output': placed[name].span(base.data_start)}
            if group == 'H1':
                item['expected_payload_sha256'] = recon['H1_payload_sha256'][name]
            tensors.append(item)
        groups[group] = {'new_tensor_count': len(names), 'total_tensor_count_replaced': len(edits),
                         'payload_delta_bytes': delta, 'output_file_size': end, 'tensors': tensors}

    return {'schema_version': 1, 'sources': {label: {
                k: pins[label][k] for k in ('repo', 'revision', 'filename', 'size_bytes', 'advertised_sha256')
            } for label in ('H0', 'U')}, 'header_range_bytes': HEADER_BYTES, 'header_sha256': digests,
            'data_start': {'H0': base.data_start, 'U': donor.data_start},
            'tensor_counts': {'H0': len(base.tensors), 'U': len(donor.tensors)},
            'compatible_metadata_sha256': {k: sha256(base.metadata[k]).hexdigest() for k in keys},
            'not_imported_U_scales': sorted(extra), 'incremental_groups': groups,
            'selected_U_payload_bytes': sum(t.nbytes for t in edits.values()),
            'H0_output_weight': base.tensors['output.weight'].span(base.data_start)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def transfer(read, start: int, size: int, write) -> str:
    """Copy one exact-sized span without allocating the tensor."""
    digest = sha256()
    stop = start + size
    for offset in range(start, stop, COPY_CHUNK):
        count = min(COPY_CHUNK, stop - offset)
        part = read(offset, count)
        if len(part) != count:
            raise ValueError(f'truncated tensor payload at absolute offset {offset}')
        digest.update(part)
        write(part)
    return digest.hexdigest()


def scan(file, header: Header, raw_header: memoryview) -> tuple[str, dict[str, str]]:
    """Hash one complete snapshot and its individual tensor spans in the same pass."""
    file.seek(0)
    disk_header = file.read(header.data_start)
    if disk_header != raw_header:
        raise ValueError('GGUF header changed before hashing')
    full = sha256(disk_header)

    def read(_, count):
        part = file.read(count)
        full.update(part)
        return part

    def discard(_):
        pass

    digests = {}
    for tensor in sorted(header.tensors.values(), key=lambda t: t.offset):
        transfer(read, 0, header.data_start + tensor.offset - file.tell(), discard)
        digests[tensor.name] = transfer(read, 0, tensor.nbytes, discard)
    transfer(read, 0, header.file_size - file.tell(), discard)
    if file.read(1):
        raise ValueError('GGUF file grew while hashing')
    return full.hexdigest(), digests


def pad(file, stop: int) -> None:
    count = stop - file.tell()
    if count < 0:
        raise ValueError('GGUF layout exceeded predicted offset')
    while count:
        part = min(count, COPY_CHUNK)
        file.write(bytes(part))
        count -= part


def write_copy(base_file, base: Header, edits: dict[str, Tensor], donor_start: int, donor_read,
               expected_base: dict[str, str], expected_donor: dict[str, str], out) -> dict:
    placed, end = layout(base, edits)
    out.write(struct.pack('<4sIQQ', b'GGUF', 3, len(placed), len(base.metadata)))
    out.write(base.metadata_block)
    for name in base.tensors:  # preserve descriptor order; payload order follows offsets
        tensor = placed[name]
        raw_name = name.encode('utf-8')
        out.write(struct.pack('<Q', len(raw_name)))
        out.write(raw_name)
        out.write(struct.pack(f'<I{len(tensor.shape)}QIQ', len(tensor.shape), *tensor.shape,
                              tensor.type_id, tensor.offset))
    digests = {}

    def base_read(offset, count):
        return os.pread(base_file.fileno(), count, offset)

    for name, tensor in placed.items():
        pad(out, base.data_start + tensor.offset)
        old = base.tensors[name]
        if name in edits:
            source = edits[name]
            got = transfer(donor_read, donor_start + source.offset, source.nbytes, out.write)
            expected = expected_donor.get(name)
        else:
            got = transfer(base_read, base.data_start + old.offset, old.nbytes, out.write)
            expected = expected_base.get(name)
        if expected and got != expected:
            raise ValueError(f'{name}: copied payload hash mismatch')
        digests[name] = got
    pad(out, end)
    return digests


class IndependentHeader(GGUFFile):
    """Use the project's independent decoder without letting it read beyond checked header bytes."""

    def __init__(self, path: Path, raw: memoryview):
        self.path = path
        self.metadata = {}
        self.tensors = []
        self.alignment = 32
        self._parse(io.BytesIO(raw))
        self.data_start = self._data_start


def verify_copy(path: Path, base: Header, edits: dict[str, Tensor], digests: dict[str, str]) -> str:
    placed, end = layout(base, edits)
    with path.open('rb') as file:
        size = os.fstat(file.fileno()).st_size
        raw = file.read(HEADER_BYTES)
        actual = parse_header(raw, size)
        if size != end or actual.metadata_block != base.metadata_block:
            raise ValueError('output size or typed metadata differs from the planned layout')
        raw_header = memoryview(raw)[:actual.data_start]
        independent = IndependentHeader(path, raw_header)
        independent_desc = [(t.name, tuple(t.shape), t.type_id, t.offset) for t in independent.tensors]
        expected_desc = [(n, placed[n].shape, placed[n].type_id, placed[n].offset) for n in base.tensors]
        if independent.data_start != base.data_start or independent_desc != expected_desc:
            raise ValueError('independent reader found different output tensor descriptors')
        full, reread = scan(file, actual, raw_header)
        for name, digest in reread.items():
            if digest != digests[name]:
                raise ValueError(f'{name}: output payload hash mismatch')
        return full


def check_destinations(output: Path, manifest: Path) -> None:
    if output.resolve() == manifest.resolve() or os.path.lexists(output) or os.path.lexists(manifest):
        raise FileExistsError('GGUF and manifest must be distinct, unused paths')


def write_verified(base_file, base: Header, edits: dict[str, Tensor], donor_start: int, donor_read,
                   expected_base: dict[str, str], expected_donor: dict[str, str], output: Path,
                   manifest: Path, provenance: dict) -> dict:
    # An atomic hard-link publication provides rename without clobbering an existing destination.
    # Temporary files are created beside their destination; neither input is ever opened writable.
    check_destinations(output, manifest)
    with tempfile.NamedTemporaryFile(mode='wb', dir=output.parent, prefix='.inkling-copy-') as staged:
        digests = write_copy(base_file, base, edits, donor_start, donor_read, expected_base, expected_donor, staged)
        staged.flush()
        os.fsync(staged.fileno())
        full_hash = verify_copy(Path(staged.name), base, edits, digests)
        result = {**provenance, 'file': str(output.resolve()), 'size_bytes': os.fstat(staged.fileno()).st_size,
                  'sha256': full_hash, 'unchanged_payload_count': len(base.tensors) - len(edits),
                  'typed_metadata_sha256': sha256(base.metadata_block).hexdigest(),
                  'replacements': list(edits), 'payload_sha256': digests}
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=manifest.parent,
                                         prefix='.inkling-manifest-') as note:
            json.dump(result, note, indent=2)
            note.write('\n')
            note.flush()
            os.fsync(note.fileno())
            os.link(note.name, manifest)
            try:
                os.link(staged.name, output)
            except OSError as exc:
                # Public paths can change concurrently. Never unlink one to roll back a collision.
                raise OSError(f'GGUF not published: {exc}; detached manifest left at {manifest}') from exc
    return result


def prepare_a1(base: Header, pin: dict, token: bytes):
    inventory = read_json(RECORDS / 'orca-inventory.json')
    if any(pin[k] != inventory['source'][k] for k in DONOR_KEYS):
        raise ValueError('Orca donor pin differs from the A0 inventory')
    plan = inventory['A1']
    if set(plan['tensors']) != set(A1_NAMES) or plan['required_scales'] != []:
        raise ValueError('A1 requires exactly 24 attention output tensors and no scales')
    remote = PublicRange(pin['repo'], pin['revision'], pin['filename'], pin['size_bytes'],
                         advertised_sha256=pin['advertised_sha256'], token=token)
    raw = remote.read(0, HEADER_BYTES)
    probes = inventory['access']
    header_sha = sha256(raw).hexdigest()
    if header_sha != probes['header_range']['sha256']:
        raise ValueError('Orca header differs from the A0 pin')
    donor = parse_header(raw, remote.size)
    if donor.data_start != inventory['header']['data_start']:
        raise ValueError('Orca data offset differs from A0')
    edits = {}
    for name in A1_NAMES:
        old, new = base.tensors[name], donor.tensors[name]
        if (old.type_id, new.type_id, old.shape, new.shape, new.nbytes) != (2, 8, (3072, 2304), (3072, 2304), 7520256):
            raise ValueError(f'{name}: expected Q4_0 → Q8_0 with exact A1 dimensions')
        if new.span(donor.data_start) | {'shape': list(new.shape)} != plan['tensors'][name]:
            raise ValueError(f'{name}: donor span differs from A0')
        edits[name] = new
    if len(base.tensors) != 704 or layout(base, edits)[1] != 15467190976:
        raise ValueError('A1 candidate size or tensor count differs from A0')
    payload_probe = probes['one_payload_range']
    if payload_probe['name'] != A1_NAMES[0]:
        raise ValueError('missing A0 payload hash for the first A1 tensor')
    observation = {'method': 'authorized bounded ranges', 'header_sha256': header_sha, 'verified_file_sha256': None}
    return edits, donor.data_start, remote.read, {A1_NAMES[0]: payload_probe['sha256']}, observation


def build(args: argparse.Namespace) -> dict:
    check_destinations(args.output, args.manifest)
    pins = read_json(args.lock)['sources']
    edits, expected_donor = {}, {}
    donor_start, donor_read = 0, None
    provenance = {'schema_version': 1, 'variant': args.variant}
    with args.base.open('rb') as file:
        size = os.fstat(file.fileno()).st_size
        raw = file.read(HEADER_BYTES)
        base = parse_header(raw, size)
        full_hash, expected_base = scan(file, base, memoryview(raw)[:base.data_start])
        if args.variant == 'H0-noop':
            if args.parent_manifest:
                raise ValueError('no-op takes only the pinned H0 as parent')
            if size != pins['H0']['size_bytes'] or full_hash != pins['H0']['advertised_sha256']:
                raise ValueError('no-op parent must match the full pinned H0 hash and size')
        else:
            if not args.parent_manifest:
                raise ValueError(f'{args.variant} needs --parent-manifest')
            parent = read_json(args.parent_manifest)
            required = {'H1': 'H0-noop', 'H2': 'H1', 'H3': 'H2', 'A1': 'H3'}[args.variant]
            if (parent['variant'], parent['sha256'], parent['size_bytes']) != (required, full_hash, size):
                raise ValueError(f'parent GGUF does not match the required {required} manifest')
            if expected_base != parent['payload_sha256']:
                raise ValueError('parent payload hashes differ from its manifest')
            if args.variant == 'A1':
                token = sys.stdin.buffer.readline(512).removesuffix(b'\n')
                edits, donor_start, donor_read, expected_donor, observation = prepare_a1(base, pins['A'], token)
                provenance.update(parent_sha256=full_hash, donor={k: pins['A'][k] for k in DONOR_KEYS},
                                  donor_verification=observation)
                return write_verified(file, base, edits, donor_start, donor_read, expected_base, expected_donor,
                                      args.output, args.manifest, provenance)
            plan = read_json(args.plan)
            names = GROUPS[args.variant]
            entries = plan['incremental_groups'][args.variant]['tensors']
            if [item['name'] for item in entries] != list(names):
                raise ValueError('transplant plan must match the stage whitelist exactly and in order')
            remote = donor_range(pins['U'])
            donor_start = plan['data_start']['U']
            donor_read = remote.read
            raw_u = donor_read(0, HEADER_BYTES)
            recon = read_json(args.recon)
            if sha256(raw_u).hexdigest() != recon['header_sha256']['U']:
                raise ValueError('U header does not match the RECON pin')
            donor = parse_header(raw_u, remote.size)
            if donor_start != donor.data_start:
                raise ValueError('U data offset does not match the validated header')
            expected_h1 = recon['H1_payload_sha256']
            for item in entries:
                name = item['name']
                tensor = donor.tensors[name]
                actual_span = tensor.span(donor_start) | {'shape': list(tensor.shape)}
                if item['U'] != actual_span:
                    raise ValueError(f'{name}: donor span differed from the validated plan')
                if args.variant == 'H1':
                    if expected_base[name] != expected_h1[name]['H0']:
                        raise ValueError(f'{name}: H0 payload hash differs from RECON')
                    expected_donor[name] = expected_h1[name]['U']
                edits[name] = tensor
            planned = plan['incremental_groups'][args.variant]['output_file_size']
            if layout(base, edits)[1] != planned:
                raise ValueError('candidate size differs from the validated plan')
            provenance['donor'] = {k: pins['U'][k] for k in DONOR_KEYS}
        provenance['parent_sha256'] = full_hash
        return write_verified(file, base, edits, donor_start, donor_read, expected_base, expected_donor,
                              args.output, args.manifest, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory', help='read only the local H0 and pinned U headers')
    inv.add_argument('--base', type=Path, required=True)
    inv.add_argument('--lock', type=Path, default=RECORDS / 'experiment-lock.json')
    inv.add_argument('--recon', type=Path, default=RECORDS / 'recon-pins.json')
    copy = sub.add_parser('build', help='write and independently verify a separate GGUF')
    copy.add_argument('--variant', choices=['H0-noop', *GROUPS, 'A1'], required=True,
                      help='A1 requires an authorized Hugging Face token via stdin')
    copy.add_argument('--base', type=Path, required=True)
    copy.add_argument('--output', type=Path, required=True)
    copy.add_argument('--manifest', type=Path, required=True)
    copy.add_argument('--parent-manifest', type=Path)
    copy.add_argument('--plan', type=Path, default=RECORDS / 'transplant-plan.json')
    copy.add_argument('--lock', type=Path, default=RECORDS / 'experiment-lock.json')
    copy.add_argument('--recon', type=Path, default=RECORDS / 'recon-pins.json')
    args = parser.parse_args()
    try:
        if args.command == 'inventory':
            result = inventory(args.base, read_json(args.lock)['sources'], read_json(args.recon))
        else:
            result = build(args)
    except (OSError, ValueError, KeyError, struct.error, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')
    print(json.dumps(result if args.command == 'inventory' else {k: v for k, v in result.items()
                                                                 if k != 'payload_sha256'}, indent=2))


if __name__ == '__main__':
    main()
