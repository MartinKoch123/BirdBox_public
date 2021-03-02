import unittest
from birdbox.tools import Rect, crop_to_square_crop


class RectTest(unittest.TestCase):

    def test_attributes(self):
        r = Rect(1, 2, 3, 4)
        self.assertEqual(1, r.left)
        self.assertEqual(2, r.top)
        self.assertEqual(3, r.right)
        self.assertEqual(4, r.bottom)

    def test_tuple(self):
        r = Rect(1, 2, 3, 4)
        self.assertTupleEqual((1, 2, 3, 4), r.tuple())

    def test_area(self):
        r = Rect(1, 2, 4, 8)
        self.assertEqual(18, r.area())

    def test_size(self):
        r = Rect(-1, 2, 2, 6)
        self.assertTupleEqual((3, 4), r.size())

    def test_width(self):
        r = Rect(1, 2, 4, 8)
        self.assertEqual(3, r.width())

    def test_height(self):
        r = Rect(1, 2, 4, 8)
        self.assertEqual(6, r.height())

    def test_center(self):
        r = Rect(1, 2, 4, 8)
        self.assertTupleEqual((2.5, 5), r.center())

    def test_horizontal_center(self):
        r = Rect(1, 2, 4, 8)
        self.assertEqual(2.5, r.horizontal_center())

    def test_vertical_center(self):
        r = Rect(1, 2, 4, 8)
        self.assertEqual(5, r.vertical_center())

    def test_left_top_width_height(self):
        r = Rect(1, 2, 4, 8)
        self.assertTupleEqual((1, 2, 3, 6), r.left_top_width_height())

    def test_scale_size(self):
        r = Rect(0, 2, 4, 8)
        self.assertEqual(Rect(-2, -1, 6, 11), r.scale_size(2))

    def test_int(self):
        r = Rect(1.1, 2.2, 3.3, 4.4).int()
        self.assertIsInstance(r.top, int)
        self.assertIsInstance(r.left, int)
        self.assertIsInstance(r.bottom, int)
        self.assertIsInstance(r.right, int)

    def test_round(self):
        r = Rect(1, 2.2, 3.33, 4.444)
        self.assertEqual(Rect(1, 2.2, 3.3, 4.4), round(r, 1))

    def test_mul(self):
        actual = Rect(1, 2, 4, 8) * 2
        expected = Rect(2, 4, 8, 16)
        self.assertEqual(actual, expected)
        actual = Rect(1, 2, 4, 8) * (-1, 2)
        expected = Rect(-1, 4, -4, 16)
        self.assertEqual(actual, expected)

    def test_truediv(self):
        actual = Rect(1, 2, 4, 8) / 2
        expected = Rect(0.5, 1, 2, 4)
        self.assertEqual(actual, expected)
        actual = Rect(1, 2, 4, 8) / (2, -1)
        expected = Rect(0.5, -2, 2, -8)
        self.assertEqual(actual, expected)

    def test_eq(self):
        r1 = Rect(1, 2, 3, 4)
        r2 = Rect(1, 2, 3, 4)
        r3 = Rect(1, 2, 3, 5)
        self.assertTrue(r1 == r2)
        self.assertFalse(r1 == r3)

    def test_ne(self):
        r1 = Rect(1, 2, 3, 4)
        r2 = Rect(1, 2, 3, 4)
        r3 = Rect(1, 2, 3, 5)
        self.assertFalse(r1 != r2)
        self.assertTrue(r1 != r3)

    def test_intersection(self):
        r1 = Rect(1, 1, 3, 3)
        r2 = Rect(3, 1, 5, 3)
        r3 = Rect(2, 2, 5, 4)
        self.assertEqual(0, Rect.intersection(r1, r2))
        self.assertEqual(4, Rect.intersection(r1, r1))
        self.assertEqual(1, Rect.intersection(r1, r3))

    def test_union(self):
        r1 = Rect(1, 1, 3, 3)
        r2 = Rect(3, 1, 5, 3)
        r3 = Rect(2, 2, 5, 4)
        self.assertEqual(8, Rect.union(r1, r2))
        self.assertEqual(4, Rect.union(r1, r1))
        self.assertEqual(9, Rect.union(r1, r3))

    def test_intersection_over_union(self):
        r1 = Rect(1, 1, 3, 3)
        r2 = Rect(2, 2, 5, 4)
        self.assertEqual(1/9, Rect.intersection_over_union(r1, r2))

    def test_from_top_left_width_height(self):
        actual = Rect.from_left_top_width_height(1, 2, 3, 4)
        expected = Rect(1, 2, 4, 6)
        self.assertEqual(actual, expected)


class CropToSquareCropTest(unittest.TestCase):
    def test_square_input(self):
        image_size = 3, 3
        crop = Rect(0, 0, 1, 1)
        actual = crop_to_square_crop(crop, image_size)
        expected = crop
        self.assertEqual(actual, expected)

    def test_unconstrained(self):
        image_size = 5, 5
        crop = Rect(2, 1, 3, 4)
        actual = crop_to_square_crop(crop, image_size)
        expected = Rect(1, 1, 4, 4)
        self.assertEqual(actual, expected)

    def test_constrained(self):
        image_size = 5, 5
        crop = Rect(0, 0, 1, 2)
        actual = crop_to_square_crop(crop, image_size)
        expected = Rect(0, 0, 2, 2)
        self.assertEqual(actual, expected)

    def test_image_too_small(self):
        image_size = 2, 5
        crop = Rect(0, 0, 1, 4)
        actual = crop_to_square_crop(crop, image_size)
        expected = Rect(0, 1, 2, 3)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
