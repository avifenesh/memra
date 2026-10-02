import unittest
from prometheus_metrics import histogram, parse_samples, scalar


class PrometheusSamples(unittest.TestCase):
    def test_labels_are_unescaped_and_order_independent(self):
        samples = parse_samples('m{lane="interactive",model="a\\\"b\\\\c\\n"} 12\n')
        self.assertEqual(scalar(samples, 'm', model='a"b\\c\n', lane='interactive'), 12)

    def test_duplicate_series_or_label_is_rejected(self):
        for text in ('m 1\nm 2\n', 'm{x="a",x="b"} 1\n', 'm{x="a",} 1\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_samples(text)

    def test_nonfinite_or_malformed_samples_are_rejected(self):
        for text in ('m NaN', 'm +Inf', 'm 1 garbage', 'm{x="\\t"} 1'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_samples(text)

    def test_cumulative_histogram(self):
        samples = parse_samples('h_bucket{model="m",le="0.5"} 1\nh_bucket{model="m",le="+Inf"} 2\nh_count{model="m"} 2\nh_sum{model="m"} 1.2\n')
        self.assertEqual(histogram(samples, 'h', model='m')['count'], 2)

    def test_bad_histogram_counts_are_rejected(self):
        template='h_bucket{le="0.5"} VALUE\nh_bucket{le="+Inf"} 2\nh_count 2\nh_sum 1.2\n'
        for value in ('3', '-1', '1.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                histogram(parse_samples(template.replace('VALUE', value)), 'h')

    def test_invalid_numeric_bucket_bounds_are_rejected(self):
        for bound in ('-Inf', 'NaN', '-1'):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                histogram(parse_samples('h_bucket{le="' + bound + '"} 1\nh_count 1\nh_sum 0.2'), 'h')

    def test_missing_infinite_bucket_is_rejected(self):
        with self.assertRaises(ValueError):
            histogram(parse_samples('h_bucket{le="0.5"} 1\nh_count 1\nh_sum 0.2'), 'h')


if __name__ == '__main__':
    unittest.main()
