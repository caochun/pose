"""Profile naming checks; no Hermes installation or network access needed."""

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProfileNamesTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "pose_queries" / "labels.py"
        spec = importlib.util.spec_from_file_location("profile_names_test", source)
        self.labels = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.labels)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = Path(folder.name) / "profiles.sqlite3"
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY)")
            connection.executemany("INSERT INTO profiles VALUES(?)", [(3,), (4,)])
        override = patch.object(self.labels, "DB", self.db)
        override.start()
        self.addCleanup(override.stop)

    def test_name_persists_and_resolves_after_rename(self):
        self.assertIn("已命名", self.labels.rename("P3 阿春"))
        self.assertEqual(self.labels.resolve("阿春"), 3)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute("SELECT name FROM profile_names WHERE profile_id=3").fetchone(), ("阿春",))
        self.labels.rename("P3 Chun")
        self.assertEqual(self.labels.resolve("chun"), 3)
        self.assertEqual(self.labels.decorate_events("P3（轨迹T1）出现"), "Chun（P3）（轨迹T1）出现")

    def test_invalid_or_duplicate_names_preserve_existing_binding(self):
        self.labels.rename("P3 阿春")
        self.assertIn("已绑定", self.labels.rename("P4 阿春"))
        self.assertIn("不存在", self.labels.rename("P999 无人"))
        self.assertIn("限1", self.labels.rename("P3 MEDIA:/root/.env"))
        self.assertEqual(self.labels.names(), {3: "阿春"})

    def test_clear_preserves_profile_records(self):
        self.labels.rename("P3 阿春")
        self.assertIn("已清除", self.labels.rename("P3 -"))
        self.assertEqual(self.labels.names(), {})
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
