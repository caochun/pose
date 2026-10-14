"""Protect coordinate conventions and track continuity at the model boundary."""
import unittest

import numpy as np

from motionbert_input import coco2h36m, prepare_input
from run_motionbert import make_windows


class MotionBERTInputTests(unittest.TestCase):
    def test_joint_semantics(self):
        points = np.arange(17 * 2, dtype=np.float32).reshape(1, 17, 2)
        mapped = coco2h36m(points)
        np.testing.assert_array_equal(mapped[:, 1], points[:, 12])  # right hip
        np.testing.assert_array_equal(mapped[:, 16], points[:, 10])  # right wrist
        np.testing.assert_array_equal(mapped[:, 0], points[:, [11, 12]].mean(axis=1))
        np.testing.assert_array_equal(mapped[:, 8], points[:, [5, 6]].mean(axis=1))
        np.testing.assert_array_equal(mapped[:, 7], mapped[:, [0, 8]].mean(axis=1))

    def test_translation_and_scale_invariance(self):
        xy = np.random.default_rng(3).uniform(100, 500, (8, 17, 2)).astype(np.float32)
        conf = np.ones((8, 17), dtype=np.float32)
        actual = prepare_input(xy, conf, (1280, 720))
        translated = prepare_input(xy + [100, 70], conf, (1280, 720))
        scaled = prepare_input(xy * 2, conf, (2560, 1440))
        # The padded fake person uses the official pre-scaling convention;
        # only observed-person coordinates should be translation invariant.
        np.testing.assert_allclose(actual[:, 0], translated[:, 0], atol=1e-6)
        np.testing.assert_allclose(actual, scaled, atol=1e-6)
        self.assertEqual(actual.shape, (1, 2, 32, 17, 3))

    def test_invalid_clip_rejected(self):
        xy = np.zeros((8, 17, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            prepare_input(xy, np.zeros((8, 17)), (1280, 720))
        xy[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_input(xy, np.ones((8, 17)), (1280, 720))

    def snapshot(self, stamp, pid=1, tids=(1,)):
        return {"timestamp": stamp, "monitor_pid": pid, "frame_size": [1280, 720],
                "people": [{"track_id": t, "profile_id": t, "confidence": [1] * 17}
                           for t in tids]}

    def test_people_are_never_mixed(self):
        snapshots = [self.snapshot(t, tids=(1, 2)) for t in range(3)]
        windows = list(make_windows(snapshots, seconds=2, min_samples=3))
        self.assertEqual({key for key, clip in windows}, {(1, 1), (2, 2)})
        for key, clip in windows:
            self.assertTrue(all(row[1]["track_id"] == key[0] for row in clip))

    def test_absence_restart_and_gap_reset_history(self):
        baseline = [self.snapshot(0), self.snapshot(0.5), self.snapshot(0.95)]
        self.assertEqual(len(list(make_windows(baseline, seconds=1.2, min_samples=3))), 1)
        absent = baseline[:2] + [self.snapshot(0.6, tids=())] + baseline[2:]
        restarted = baseline[:2] + [self.snapshot(0.95, pid=2)]
        for snapshots in (absent, restarted):
            self.assertEqual(list(make_windows(snapshots, seconds=1.2, min_samples=3)), [])
        gap = [self.snapshot(0), self.snapshot(1), self.snapshot(4)]
        self.assertEqual(list(make_windows(gap, seconds=5, min_samples=3)), [])


if __name__ == "__main__":
    unittest.main()
