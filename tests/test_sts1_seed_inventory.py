from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.sts1.sts1_g7_seed_inventory import (
    _parse_seed_field_candidates,
    _parse_seed_list,
    _read_private_source,
    write_inventory,
)


class SeedInventoryTests(unittest.TestCase):
    def test_seed_list_parser_accepts_only_numeric_rows(self) -> None:
        self.assertEqual(_parse_seed_list("# header\n7\n11, 13\n", source="fixture"), [7, 11, 13])
        with self.assertRaises(ValueError):
            _parse_seed_list("7\nwin=true\n", source="fixture")

    def test_seed_field_parser_ignores_non_seed_fields(self) -> None:
        text = "seed_ids: [17, 19]\nrng_seed: 23\nseed_count: 2\nunrelated: 997\n# seed: 89\n"
        self.assertEqual(
            _parse_seed_field_candidates(text, source="fixture"),
            [17, 19, 23, 2],
        )

    def test_private_source_loader_requires_hash_and_numeric_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "sts1-private-seed-source-v1",
                        "category": "reserved",
                        "source_ref": "fixture/ids.txt",
                        "sha256": "a" * 64,
                        "seed_ids": [29, 31],
                    }
                ),
                encoding="utf-8",
            )
            source = _read_private_source(path)
            self.assertEqual(source["seed_ids"], [29, 31])

    def test_private_inventory_must_stay_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            private_path = root / "private.json"
            public_path = root / "audit.json"
            with self.assertRaises(ValueError):
                write_inventory(
                    {},
                    {},
                    private_path=private_path,
                    audit_path=public_path,
                    repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
