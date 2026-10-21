"""Keep grid compaction within its scheduling and UB resource contract."""
import unittest

from code_dsl.op_host.block_schedule import select_blocks


class BlockScheduleTests(unittest.TestCase):
    def test_compact_equal_round_grids(self):
        for available in (20, 24):
            with self.subTest(available=available):
                self.assertEqual(select_blocks(256, 576, 4, 96, available), 16)

    def test_keep_grid_when_wider_rows_exceed_worker_ub(self):
        self.assertEqual(select_blocks(256, 1100, 4, 96, 20), 20)

    def test_keep_other_numerical_strategies(self):
        for args in ((128, 264, 4, 48, 20), (200, 416, 3, 0, 18),
                     (256, 576, 1, 96, 20)):
            with self.subTest(args=args):
                self.assertEqual(select_blocks(*args), args[-1])
