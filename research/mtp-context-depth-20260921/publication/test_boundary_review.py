import copy
import json
import tempfile
import unittest
from pathlib import Path
from verify_boundary import verify,store,ARCHIVES


def fixture():
    scan={'runtime_archive_sha256':'source','scanner_sha256':'scanner','policy_sha256':'policy',
          'archives':[{'file':n,'sha256':n,'files':1,'compressed_rules':[], 'expanded_matches':[]}
                      for n in sorted(ARCHIVES)]}
    review={k:scan[k] for k in ('runtime_archive_sha256','scanner_sha256','policy_sha256')}
    review.update(archive_inventory=[{k:r[k] for k in ('file','sha256','files','compressed_rules')}
                                     for r in scan['archives']],source_rule_pins=[],record_rule_pins=[])
    return scan,review


class BoundaryReviewTests(unittest.TestCase):
    def test_complete_inventory_passes(self):
        scan,review=fixture();self.assertEqual(verify(scan,review)['unresolved_matches'],0)

    def test_removing_an_archive_cannot_shrink_the_review(self):
        scan,review=fixture();scan['archives'].pop();review['archive_inventory'].pop()
        with self.assertRaisesRegex(ValueError,'four-archive'):verify(scan,review)

    def test_changed_expanded_file_needs_a_new_exact_decision(self):
        scan,review=fixture()
        scan['archives'][0]['expanded_matches']=[{'file':'x','sha256':'changed','rules':['rule']}]
        review['record_rule_pins']=[{'archive':scan['archives'][0]['file'],'file':'x','sha256':'old','rules':['rule']}]
        with self.assertRaisesRegex(ValueError,'exact reviewed'):verify(scan,review)

    def test_unused_pin_is_rejected(self):
        scan,review=fixture();review['source_rule_pins']=[{'file':'x','sha256':'x','rules':['rule']}]
        with self.assertRaisesRegex(ValueError,'not exercised'):verify(scan,review)

    def test_check_does_not_rewrite_the_sealed_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json';path.write_text('{"original":true}\n');before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'differs'):store(path,{'other':True},True)
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
