from __future__ import absolute_import, unicode_literals

import numpy as np
import pytest

from spacy import attrs

from textacy import cache, export


@pytest.fixture(scope='module')
def spacy_doc():
    text = "I would have lived in peace. But my enemies brought me war."
    spacy_lang = cache.load_spacy('en')
    spacy_doc = spacy_lang(text)
    cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
    values = np.array(
        [[13656873538139661788, 3, 426],
         [16235386156175103506, 2, 402],
         [14200088355797579614, 1, 402],
         [3822385049556375858, 0, 8206900633647566924],
         [1292078113972184607, 18446744073709551615, 440],
         [15308085513773655218, 18446744073709551615, 436],
         [12646065887601541794, 18446744073709551613, 442],
         [17571114184892886314, 3, 404],
         [4062917326063685704, 1, 437],
         [783433942507015291, 1, 426],
         [17109001835818727656, 0, 8206900633647566924],
         [13656873538139661788, 18446744073709551615, 3965108062993911700],
         [15308085513773655218, 18446744073709551614, 413],
         [12646065887601541794, 18446744073709551613, 442]],
        dtype='uint64')
    spacy_doc.from_array(cols, values)
    return spacy_doc


def test_write_conll(spacy_doc):
    expected = '# sent_id 1\n1\tI\t-PRON-\tPRON\tPRP\t_\t4\tnsubj\t_\t_\n2\twould\twould\tVERB\tMD\t_\t4\taux\t_\t_\n3\thave\thave\tVERB\tVB\t_\t4\taux\t_\t_\n4\tlived\tlive\tVERB\tVBN\t_\t0\troot\t_\t_\n5\tin\tin\tADP\tIN\t_\t4\tprep\t_\t_\n6\tpeace\tpeace\tNOUN\tNN\t_\t5\tpobj\t_\tSpaceAfter=No\n7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_\n\n# sent_id 2\n1\tBut\tbut\tCCONJ\tCC\t_\t4\tcc\t_\t_\n2\tmy\t-PRON-\tADJ\tPRP$\t_\t3\tposs\t_\t_\n3\tenemies\tenemy\tNOUN\tNNS\t_\t4\tnsubj\t_\t_\n4\tbrought\tbring\tVERB\tVBD\t_\t0\troot\t_\t_\n5\tme\t-PRON-\tPRON\tPRP\t_\t4\tdative\t_\t_\n6\twar\twar\tNOUN\tNN\t_\t4\tdobj\t_\tSpaceAfter=No\n7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tSpaceAfter=No\n'
    observed = export.doc_to_conll(spacy_doc)
    assert observed == expected
