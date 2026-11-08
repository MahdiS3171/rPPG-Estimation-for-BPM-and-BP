import unittest
from rppg_lab.waveform_research_split import grouped_partition,validate_partition


class SplitTests(unittest.TestCase):
    def test_complete_identity_disjoint_partition_is_order_independent(self):
        ids=[f'subject{i}' for i in range(41)]
        a=grouped_partition(ids,(29,6,6),20261110)
        self.assertEqual(a,grouped_partition(list(reversed(ids)),(29,6,6),20261110))
        validate_partition(a,ids)
        self.assertEqual([len(a[k]) for k in ('train','validation','test')],[29,6,6])

    def test_duplicate_or_cross_role_subject_fails(self):
        with self.assertRaises(ValueError):grouped_partition(['a','a','b'],(1,1,1),1)
        with self.assertRaises(ValueError):validate_partition(dict(train=['a'],validation=['b'],test=['a']),['a','b','c'])


if __name__=='__main__':unittest.main()
