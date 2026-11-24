"""Small file and HTTP-response fixtures for the Inkling raw writer (no GPU or network)."""
import hashlib
import io
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.inkling_weights import transplant as tx


def string(value):
    return struct.pack('<Q', len(value)) + value


METADATA = (string(b'general.alignment') + struct.pack('<II', 4, 64)
            + string(b'ids') + struct.pack('<IIQHH', 9, 2, 2, 1, 65535)
            + string(b'labels') + struct.pack('<IIQ', 9, 8, 2) + string(b'alpha') + string(b'beta'))


def fixture(tensors, payload):
    header = struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), 3) + METADATA
    for name, shape, kind, offset in tensors:
        header += string(name.encode()) + struct.pack(f'<I{len(shape)}QIQ', len(shape), *shape, kind, offset)
    return header + bytes(tx.align(len(header), 64) - len(header)) + payload


class TransplantTest(unittest.TestCase):
    def test_roundtrip_growing_tensor_and_typed_metadata(self):
        # Descriptors are not in payload order. A larger first payload moves the second one.
        raw = fixture([('keep', (3,), 0, 64), ('grow', (32, 2), 2, 0)],
                      b'B' * 36 + bytes(28) + b'C' * 12)
        donor = b'D' * 128
        with tempfile.TemporaryDirectory() as root:
            source, output, manifest = (Path(root) / name for name in ('input', 'output', 'manifest'))
            source.write_bytes(raw)
            header = tx.parse_header(raw, len(raw))
            expected = {'keep': hashlib.sha256(b'C' * 12).hexdigest()}
            edits = {'grow': tx.Tensor('grow', (32, 2), 30, 0)}
            with source.open('rb') as file:
                record = tx.write_verified(file, header, edits, 0, lambda o, n: donor[o:o + n],
                                           expected, {'grow': hashlib.sha256(donor).hexdigest()},
                                           output, manifest, {'variant': 'synthetic'})
                for existing, sidecar in ((source, manifest), (source, Path(root) / 'new')):
                    with self.assertRaises(FileExistsError):
                        tx.write_verified(file, header, {}, 0, None, {}, {}, existing, sidecar, {})
            actual = tx.parse_header(output.read_bytes(), output.stat().st_size)
            self.assertEqual(list(actual.tensors), ['keep', 'grow'])
            self.assertEqual(actual.tensors['keep'].offset, 128)
            self.assertEqual(actual.tensors['grow'].type_id, 30)
            self.assertEqual(actual.metadata_block, METADATA)
            self.assertEqual(record['unchanged_payload_count'], 1)
            self.assertEqual(record['payload_sha256']['keep'], expected['keep'])
            self.assertEqual(record['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(tx.read_json(manifest), record)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual({p.name for p in Path(root).iterdir()}, {'input', 'output', 'manifest'})

    def test_rejects_bad_descriptors_and_spans(self):
        cases = [
            ([('a', (1,), 0, 0), ('a', (1,), 0, 64)], 'duplicate tensor'),
            ([('a', (1,), 999, 0)], 'unsupported tensor type'),
            ([('a', (16, 2), 2, 0)], 'row geometry'),  # total divides by 32; row does not
            ([('a', (1,), 0, 1)], 'unaligned'),
            ([('a', (32,), 0, 0), ('b', (1,), 0, 64)], 'overlapping'),
            ([('a', (64,), 0, 0)], 'span exceeds'),
        ]
        for tensors, error in cases:
            with self.subTest(tensors=tensors), self.assertRaisesRegex(ValueError, error):
                raw = fixture(tensors, bytes(128))
                tx.parse_header(raw, len(raw))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            tx.parse_header(raw[:30], len(raw))

    def test_rejects_missing_tensor_and_truncated_payload(self):
        raw = fixture([('a', (1,), 0, 0)], bytes(4))
        header = tx.parse_header(raw, len(raw))
        with self.assertRaisesRegex(ValueError, 'missing tensors'):
            tx.layout(header, {'b': tx.Tensor('b', (1,), 0, 0)})
        with self.assertRaisesRegex(ValueError, 'truncated tensor payload'):
            tx.transfer(lambda _o, _n: b'123', 0, 4, io.BytesIO().write)

    def test_bad_payload_does_not_publish(self):
        raw = fixture([('a', (1,), 0, 0)], bytes(4))
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryFile() as file:
            file.write(raw)
            file.flush()
            with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
                tx.write_verified(file, tx.parse_header(raw, len(raw)), {}, 0, None, {'a': 'wrong'}, {},
                                  Path(root) / 'out', Path(root) / 'manifest', {})
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_source_change_after_pinned_scan_does_not_publish(self):
        raw = fixture([('a', (1,), 0, 0)], bytes(4))
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryFile() as file:
            file.write(raw)
            file.flush()
            base = tx.parse_header(raw, len(raw))
            digest, expected = tx.scan(file, base, memoryview(raw)[:base.data_start])
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            os.pwrite(file.fileno(), b'bad!', base.data_start)
            with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
                tx.write_verified(file, base, {}, 0, None, expected, {},
                                  Path(root) / 'out', Path(root) / 'manifest', {})
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_publish_collision_does_not_unlink_competing_file(self):
        raw = fixture([('a', (1,), 0, 0)], bytes(4))
        real_link = os.link
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryFile() as file:
            output, manifest = Path(root) / 'out', Path(root) / 'manifest'
            file.write(raw)
            file.flush()

            def race(source, destination):
                if destination == output:
                    output.write_bytes(b'belongs to someone else')
                real_link(source, destination)

            with patch.object(tx.os, 'link', side_effect=race), self.assertRaisesRegex(OSError, 'detached manifest'):
                tx.write_verified(file, tx.parse_header(raw, len(raw)), {}, 0, None, {}, {}, output, manifest, {})
            self.assertEqual(output.read_bytes(), b'belongs to someone else')
            self.assertEqual({path.name for path in Path(root).iterdir()}, {'out', 'manifest'})

    @patch.object(tx.subprocess, 'run')
    def test_http_range_requires_commit_status_span_and_length(self, run):
        rev = 'a' * 40
        source = tx.PublicRange('example/model', rev, 'model.gguf', 100)
        prelude = f'HTTP/2 302\r\nx-repo-commit: {rev}\r\n\r\n'.encode()
        reply = b'HTTP/2 206\r\nContent-Range: bytes 1-3/100\r\n\r\n'
        run.return_value = subprocess.CompletedProcess([], 0, b'abc', prelude + reply)
        self.assertEqual(source.read(1, 3), b'abc')
        self.assertIn(f'https://huggingface.co/example/model/resolve/{rev}/model.gguf', run.call_args.args[0])
        for headers, data, error in (
            (prelude + reply.replace(b'206', b'200'), b'abc', 'expected HTTP 206'),
            (reply, b'abc', 'pinned commit'),
            (prelude.replace(rev.encode(), b'b' * 40) + reply.replace(b'\r\n\r\n',
              f'\r\nx-repo-commit: {rev}\r\n\r\n'.encode()), b'abc', 'pinned commit'),
            (prelude + reply.replace(b'\r\n\r\n', b'\r\nContent-Range: bytes 2-4/100\r\n\r\n'),
             b'abc', 'Content-Range'),
            (prelude + reply.replace(b'1-3', b'0-2'), b'abc', 'Content-Range'),
            (prelude + reply.replace(b'/100', b'/99'), b'abc', 'Content-Range'),
            (prelude + reply, b'ab', 'response length'),
        ):
            with self.subTest(headers=headers, data=data), self.assertRaisesRegex(ValueError, error):
                run.return_value = subprocess.CompletedProcess([], 0, data, headers)
                source.read(1, 3)

    @patch.object(tx.subprocess, 'run')
    def test_authorized_range_checks_origin_digest_and_keeps_token_off_argv(self, run):
        rev, digest, token = 'a' * 40, 'b' * 64, b'hf_example'
        source = tx.PublicRange('example/model', rev, 'model.gguf', 100, digest, token)
        commit = f'HTTP/2 302\r\nx-repo-commit: {rev}\r\n'.encode()
        lfs = f'x-linked-size: 100\r\nx-linked-etag: "{digest}"\r\n'.encode()
        reply = b'HTTP/2 206\r\nContent-Range: bytes 1-3/100\r\n'
        run.return_value = subprocess.CompletedProcess([], 0, b'abc', commit + lfs + b'\r\n' + reply + b'\r\n')
        self.assertEqual(source.read(1, 3), b'abc')
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--header') + 1], '@-')
        self.assertNotIn(token.decode(), ' '.join(command))
        self.assertNotIn('--location-trusted', command)
        self.assertNotIn(token.decode(), repr(source))
        self.assertEqual(run.call_args.kwargs['input'], b'Authorization: Bearer ' + token + b'\n')
        run.return_value = subprocess.CompletedProcess([], 0, b'abc', commit + b'\r\n' + reply + lfs + b'\r\n')
        with self.assertRaisesRegex(ValueError, 'origin did not confirm'):
            source.read(1, 3)


if __name__ == '__main__':
    unittest.main()
