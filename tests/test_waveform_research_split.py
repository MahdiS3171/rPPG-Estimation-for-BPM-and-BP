import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from rppg_lab.artifacts import file_sha256
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

    def test_development_loader_refuses_test_file_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);partitions={}
            for role in ('train','validation'):
                path=root/(role+'_arrays.npz')
                np.savez(path,x=np.zeros(1),record_id=np.asarray([role]),validation=np.asarray([role=='validation']))
                partitions[role]=dict(path=path.name,sha256=file_sha256(path))
            (root/'data_receipt.json').write_text(json.dumps(dict(partitions=partitions)))
            (root/'test_arrays.npz').write_bytes(b'closed')
            code="from pathlib import Path; import sys; from rppg_lab.waveform_research_split import load_development_arrays; load_development_arrays(sys.argv[1]);\ntry: (Path(sys.argv[1])/'test_arrays.npz').read_bytes()\nexcept PermissionError: print('closed')\nelse: raise AssertionError('Test access was allowed')"
            result=subprocess.run([sys.executable,'-c',code,folder],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),'closed')

    def test_registry_and_growth_controls_keep_roles_separate(self):
        from scripts.run_reset_waveform_campaign import registry,scope_indices
        configs=registry();self.assertEqual(len(configs),40);self.assertEqual(len({c['name'] for c in configs}),40)
        ids=np.asarray([f'person{i}' for i in range(29) for _ in range(10)]+['synthetic']*2+['validation'])
        source=np.asarray([0]*290+[1]*2+[0]);starts=np.asarray(list(range(0,20,2))*29+[0,10,0],float)
        arrays=dict(record_id=ids,source=source,start=starts);fit=np.arange(292)
        small=scope_indices(arrays,fit,'21_real_subjects');sparse=scope_indices(arrays,fit,'nonoverlapping_windows')
        self.assertEqual(len(set(ids[small][source[small]==0])),21)
        self.assertEqual(len(sparse),29*2+2)
        self.assertTrue(set(small)<=set(fit));self.assertTrue(set(sparse)<=set(fit))


if __name__=='__main__':unittest.main()
