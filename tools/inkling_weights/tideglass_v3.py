"""Build the compact Tideglass v3 variant from two verified local GGUFs."""
import ctypes as ct
from dataclasses import replace
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import transplant as gg

PARENT = '7af63a2c9362f0ac6fc708443348ead5dbfcb7ecde8cd454ea3572c68864aa72'
DONOR = dict(repo='nokeepai/Inkling-2-Mini-Unbound-GGUF', revision='accd51aadc5aaef0d16f9a3e5d71f0b5dab02e9d',
             filename='Inkling-2-Mini-Unbound-v3-Q8_0.gguf', size_bytes=27820099264,
             sha256='a497670d6d9ac3be262953c0679dc064bc92ba8af198e6f7073bc3d41f7efd2a')


def plan(parent, donor):
    if len(parent.tensors) != 704 or donor.tensors.keys() != parent.tensors.keys():
        raise ValueError('unexpected tensor inventory')
    if any(t.shape != donor.tensors[n].shape for n, t in parent.tensors.items()):
        raise ValueError('donor shapes differ from parent')
    edits = {}
    for layer in range(24):
        for key in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_gate', 'attn_rel',
                    'ffn_gate_up_exps', 'ffn_down_exps'):
            name = f'blk.{layer}.{key}.weight'
            src = donor.tensors[name]
            if src.type_name != 'Q8_0':
                raise ValueError(f'{name}: expected Q8_0, found {src.type_name}')
            target = 8
            if key.startswith('ffn_'):
                down = key == 'ffn_down_exps'
                target = 3 if down and layer < 3 else 2
                if src.shape != (1152 if down else 2304, 2304, 128) or parent.tensors[name].type_id != target:
                    raise ValueError(f'{name}: unexpected expert layout')
            edits[name] = replace(src, type_id=target)
    return edits


class Quantizer:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        for name in ('dequantize_row_q8_0', 'quantize_row_q4_0_ref', 'quantize_row_q4_1_ref',
                     'dequantize_row_q4_0', 'dequantize_row_q4_1'):
            fn = getattr(self.lib, name)
            fn.restype = None
            fn.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_int64)
        self.version = {}
        for key in ('ggml_version', 'ggml_commit'):
            fn = getattr(self.lib, key)
            fn.restype = ct.c_char_p
            self.version[key] = fn().decode()
        with path.open('rb') as lib:
            self.version['library_sha256'] = file_digest(lib, 'sha256').hexdigest()

    def expert(self, data, target, elements):
        if elements % 32 or len(data) != elements // 32 * 34 or target not in ('q4_0', 'q4_1'):
            raise ValueError('invalid expert block geometry')
        floats = np.empty(elements, dtype='<f4')
        packed = np.empty(elements // 32 * (18 if target == 'q4_0' else 20), dtype='u1')
        self.lib.dequantize_row_q8_0(data, floats.ctypes.data, elements)
        if not np.isfinite(floats).all():
            raise ValueError('non-finite Q8 expert')
        getattr(self.lib, f'quantize_row_{target}_ref')(floats.ctypes.data, packed.ctypes.data, elements)
        decoded = np.empty_like(floats)
        getattr(self.lib, f'dequantize_row_{target}')(packed.ctypes.data, decoded.ctypes.data, elements)
        decoded -= floats
        if not np.isfinite(decoded).all():
            raise ValueError('non-finite quantization residual')
        return packed, float(np.max(np.abs(decoded))), float(np.sqrt(np.mean(np.square(decoded), dtype='f8')))


def inventory(f):
    size = os.fstat(f.fileno()).st_size
    raw = f.read(gg.HEADER_BYTES)
    header = gg.parse_header(raw, size)
    digest, payloads = gg.scan(f, header, memoryview(raw)[:header.data_start])
    return header, digest, payloads


def build(args):
    gg.check_destinations(args.output, args.manifest)
    quant = Quantizer(args.ggml)
    parent_note = gg.read_json(args.parent_manifest)
    with args.base.open('rb') as base, args.donor.open('rb') as src:
        parent, full, old_hash = inventory(base)
        if full != PARENT or full != parent_note['sha256'] or old_hash != parent_note['payload_sha256']:
            raise ValueError('parent or parent manifest does not match pinned v2.1')
        if os.fstat(src.fileno()).st_size != DONOR['size_bytes']:
            raise ValueError('wrong donor size')
        donor, full, src_hash = inventory(src)
        if full != DONOR['sha256']:
            raise ValueError('wrong donor SHA-256')
        edits = plan(parent, donor)
        placed, size = gg.layout(parent, edits)
        digests = old_hash.copy()
        records = {}
        with tempfile.NamedTemporaryFile(dir=args.output.parent) as out:
            gg.write_header(out, parent, placed)
            for name, t in placed.items():
                gg.pad(out, parent.data_start + t.offset)
                if name not in edits:
                    old = parent.tensors[name]
                    base.seek(parent.data_start + old.offset)
                    got = gg.transfer(lambda _, n: base.read(n), 0, old.nbytes, out.write)
                    if got != old_hash[name]:
                        raise ValueError(f'{name}: parent payload changed')
                    continue
                s = donor.tensors[name]
                offset = donor.data_start + s.offset
                src.seek(offset)
                record = dict(parent_type=parent.tensors[name].type_name, dimensions=t.shape,
                              parent_sha256=old_hash[name], donor=DONOR, source_offset=offset,
                              source_size=s.nbytes, source_type=s.type_name, source_sha256=src_hash[name],
                              target_type=t.type_name, method='copy', quantizer=None)
                if t.type_id == s.type_id:
                    got = gg.transfer(lambda _, n: src.read(n), offset, s.nbytes, out.write)
                    result = got
                else:
                    # Each contiguous slice is one whole [out,in] expert, already in GGUF
                    # order. Never transpose, split gate/up, or cross an expert boundary.
                    incoming, outgoing = sha256(), sha256()
                    rmse, maximum = [], 0.
                    for _ in range(s.shape[2]):
                        data = src.read(s.nbytes // s.shape[2])
                        incoming.update(data)
                        packed, peak, rms = quant.expert(data, t.type_name.lower(), s.shape[0] * s.shape[1])
                        out.write(packed)
                        outgoing.update(packed)
                        maximum = max(maximum, peak)
                        rmse.append(rms)
                    got, result = incoming.hexdigest(), outgoing.hexdigest()
                    record.update(method=f'dequantize_row_q8_0 -> quantize_row_{t.type_name.lower()}_ref',
                                  quantizer=quant.version, max_abs_error=maximum,
                                  expert_rmse_p0_p50_p90_p100=np.percentile(rmse, [0, 50, 90, 100]).tolist())
                if got != src_hash[name] or out.tell() != parent.data_start + t.offset + t.nbytes:
                    raise ValueError(f'{name}: donor hash or output geometry changed')
                digests[name] = record['result_sha256'] = result
                records[name] = record
            gg.pad(out, size)
            out.flush()
            os.fsync(out.fileno())
            digest = gg.verify_copy(Path(out.name), parent, edits, digests)
            manifest = dict(schema_version=1, variant='Tideglass Unbound v3', parent_sha256=PARENT,
                            donor=DONOR, file=args.output.name, sha256=digest, size_bytes=size,
                            typed_metadata_sha256=sha256(parent.metadata_block).hexdigest(),
                            precision_map={n: t.type_name for n, t in placed.items()}, storage_layout='compact',
                            unchanged_payload_count=len(placed)-len(records), replacements=records, payload_sha256=digests)
            with tempfile.NamedTemporaryFile(mode='w', dir=args.manifest.parent) as note:
                json.dump(manifest, note, indent=2, allow_nan=False)
                note.write('\n')
                note.flush()
                os.fsync(note.fileno())
                os.link(note.name, args.manifest)
                os.link(out.name, args.output)
    return {k: manifest[k] for k in ('file', 'sha256', 'size_bytes', 'unchanged_payload_count')}
