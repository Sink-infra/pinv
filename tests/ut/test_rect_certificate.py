"""Independent FP64 checks of the rectangular posterior inequalities."""

import unittest
import numpy as np


def orthogonal(rng, rows, columns):
    return np.linalg.qr(rng.normal(size=(rows, columns)), mode="reduced")[0]


def truncated_reference(a, cutoff):
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    inverse = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > cutoff)
    return (vh.T * inverse) @ u.T


class RectangularCertificateTests(unittest.TestCase):
    def test_polar_distance_is_bounded_by_the_gram_defect(self):
        rng = np.random.default_rng(308)
        q = orthogonal(rng, 9, 4)
        for scale in (1e-4, 1e-3, 1e-2):
            y = q @ (np.eye(4) + scale * rng.normal(size=(4, 4)))
            u, _, vh = np.linalg.svd(y, full_matrices=False)
            defect = np.linalg.norm(y.T @ y - np.eye(4), "fro")
            self.assertLess(defect, 0.25)
            bound = defect / (1 + np.sqrt(1 - defect))
            self.assertLessEqual(np.linalg.norm(y - u @ vh, "fro"), bound + 1e-14)

    def test_two_sided_residual_covers_the_block_diagonal_perturbation(self):
        rng = np.random.default_rng(491)
        for _ in range(6):
            u = orthogonal(rng, 8, 3)
            v = orthogonal(rng, 7, 3)
            sigma = np.diag([1.0, 0.3, 0.02])
            b = rng.normal(size=(8, 7))
            b0 = u @ sigma @ v.T + (np.eye(8) - u @ u.T) @ b @ (np.eye(7) - v @ v.T)
            first = np.linalg.norm(b @ v - u @ sigma, "fro")
            second = np.linalg.norm(b.T @ u - v @ sigma, "fro")
            self.assertLessEqual(np.linalg.norm(b - b0, "fro"), first + second + 1e-13)

    def test_separated_truncated_inverse_obeys_the_divided_difference_bound(self):
        rng = np.random.default_rng(914)
        spectrum = np.array([1.0, 0.3, 0.02, 1e-6, 0.0])
        cutoff = 1e-4
        for rows, columns in ((9, 5), (5, 9)):
            u = orthogonal(rng, rows, 5)
            v = orthogonal(rng, columns, 5)
            base = (u * spectrum) @ v.T
            expected = truncated_reference(base, cutoff)
            for scale in (1e-9, 1e-8, 1e-7):
                change = scale * rng.normal(size=base.shape)
                delta = np.linalg.norm(change, "fro")
                lower = spectrum[2] - delta
                upper = spectrum[3] + delta
                self.assertGreater(lower, cutoff)
                self.assertLess(upper, cutoff)
                actual = truncated_reference(base + change, cutoff)
                bound = delta / (lower * (lower - upper))
                self.assertLessEqual(np.linalg.norm(actual - expected, "fro"), bound + 1e-11)


if __name__ == "__main__":
    unittest.main()
