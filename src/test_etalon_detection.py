import unittest
import etalon_detection
import numpy as np


class TestExpandEdges(unittest.TestCase):
    def test_expand_edges(self):
        edges = (100, 200)
        width_multiplier = 1.5
        new_edges = etalon_detection.expand_edges(edges, width_multiplier)

        correct_width = width_multiplier * (edges[1] - edges[0])
        new_width = new_edges[1] - new_edges[0]

        self.assertEqual(new_width, correct_width)

    def test_expand_edges_zero_width(self):
        edges = (100, 100)
        width_multiplier = 1.5
        new_edges = etalon_detection.expand_edges(edges, width_multiplier)

        correct_width = width_multiplier * (edges[1] - edges[0])
        new_width = new_edges[1] - new_edges[0]

        self.assertEqual(new_width, correct_width)

    def test_expand_edges_negative_width(self):
        edges = (200, 100)
        width_multiplier = 1.5
        new_edges = etalon_detection.expand_edges(edges, width_multiplier)

        correct_width = width_multiplier * (edges[1] - edges[0])
        new_width = new_edges[1] - new_edges[0]

        self.assertEqual(new_width, correct_width)

    def test_expand_edges_zero_crossing(self):
        edges = (-100, 100)
        width_multiplier = 1.5
        new_edges = etalon_detection.expand_edges(edges, width_multiplier)

        correct_width = width_multiplier * (edges[1] - edges[0])
        new_width = new_edges[1] - new_edges[0]

        self.assertEqual(new_width, correct_width)

    def test_expand_edges_list_input(self):
        # TODO: Is this ok that the function works with lists and tuples equally?
        edges = [100, 200]
        width_multiplier = 1.5
        new_edges = etalon_detection.expand_edges(edges, width_multiplier)

        correct_width = width_multiplier * (edges[1] - edges[0])
        new_width = new_edges[1] - new_edges[0]

        self.assertEqual(new_width, correct_width)

    def test_expand_edges_return_ints(self):
        edges = (100, 200)

        width_multiplier = np.sqrt(2)
        new_edges = etalon_detection.expand_edges(edges, width_multiplier)

        self.assertIsInstance(new_edges[0], int)
        self.assertIsInstance(new_edges[1], int)


class TestMergeEdges(unittest.TestCase):
    def test_merge_edges(self):
        list_of_edges = [(199, 200), (300, 301)]
        distance_threshold = 100
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 1)
        self.assertEqual(merged_peaks[0], (199, 301))

    def test_merge_edges_no_merge(self):
        list_of_edges = [(100, 200), (300, 400)]
        distance_threshold = 50
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 2)

    def test_merge_edges_no_merge_zero_distance(self):
        list_of_edges = [(100, 200), (300, 400)]
        distance_threshold = 0
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 2)

    def test_merge_edges_no_merge_negative_distance(self):
        list_of_edges = [(100, 200), (300, 400)]
        distance_threshold = -50
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 2)

    def test_merge_edges_no_merge_zero_width(self):
        list_of_edges = [(100, 100), (300, 300)]
        distance_threshold = 50
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 2)

    def test_merge_edges_no_merge_negative_width(self):
        list_of_edges = [(200, 100), (400, 300)]
        distance_threshold = 101
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 2)

    def test_merge_edges_any_order(self):
        list_of_edges = [(300, 400), (100, 200)]
        distance_threshold = 100
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(len(merged_peaks), 1)

    def test_merge_edges_already_overlapping_zero_distance(self):
        list_of_edges = [(100, 300), (200, 400)]
        distance_threshold = 0
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(merged_peaks[0], (100, 400))
        self.assertEqual(len(merged_peaks), 1)

    def test_merge_edges_already_overlapping_with_distance(self):
        list_of_edges = [(100, 300), (200, 400)]
        distance_threshold = 200
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(merged_peaks[0], (100, 400))
        self.assertEqual(len(merged_peaks), 1)

    def test_merge_edges_same_start(self):
        list_of_edges = [(100, 200), (100, 300)]
        distance_threshold = 1
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(merged_peaks[0], (100, 300))
        self.assertEqual(len(merged_peaks), 1)

    def test_merge_edges_duplicated(self):
        list_of_edges = [(100, 200), (100, 200)]
        distance_threshold = 0
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(merged_peaks[0], (100, 200))
        self.assertEqual(len(merged_peaks), 1)



    def test_merge_edges_select_widest_overlap(self):
        list_of_edges = [(720, 724), (723, 725), (722, 728), (722, 755)] # chose from AboveNorthDock Failing at 730
        distance_threshold = 0
        merged_peaks = etalon_detection.merge_edges(list_of_edges,
                                                    distance_threshold)

        self.assertEqual(merged_peaks[0], (720, 755))
        self.assertEqual(len(merged_peaks), 1)



class TestGetPeakEdges(unittest.TestCase):
    #TODO: implement test data for this function.
    # Cases:
    # 1. No peaks
    # 2. high snr
    # 3. low snr
    def setUp(self):
        self.residual_length = 10000
        self.noise_sdev = 4e-5
        self.peak_locations = [100]
        self.peak_widths = [10]
        self.peak_amplitudes = [.005]
        self.residual = np.random.normal(scale=self.noise_sdev,
                                         size=self.residual_length)

        # add peaks based on the parameters above
        for peak_location, peak_width, peak_amplitude in zip(
                self.peak_locations, self.peak_widths, self.peak_amplitudes):
            self.residual[peak_location - peak_width:peak_location + peak_width] += peak_amplitude

    # def test_extract_noise_floor(self):
    #     noise_floor = etalon_detection.extract_noise_floor(self.residual)
    #     self.assertAlmostEqual(noise_floor, self.noise_sdev, places=4)


# class TestExtractEtalon(unittest.TestCase):

# def test_extract_etalon(self):

# Edge case
# What if the etalon_center is not in any of the notches?
