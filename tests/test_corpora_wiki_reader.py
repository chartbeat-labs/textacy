# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy.compat import unicode_type
from textacy.corpora import WikiReader
from textacy.fileio import write_file

WIKITEXT = r"""
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/ http://www.mediawiki.org/xml/export-0.10.xsd" version="0.10" xml:lang="en">
  <page>
    <title>AccessibleComputing</title>
    <ns>0</ns>
    <id>10</id>
    <redirect title="Computer accessibility" />
    <revision>
      <id>631144794</id>
      <parentid>381202555</parentid>
      <timestamp>2014-10-26T04:50:23Z</timestamp>
      <contributor>
        <username>Paine Ellsworth</username>
        <id>9092818</id>
      </contributor>
      <comment>add [[WP:RCAT|rcat]]s</comment>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text xml:space="preserve">#REDIRECT [[Computer accessibility]]

{{Redr|move|from CamelCase|up}}</text>
      <sha1>4ro7vvppa5kmm0o1egfjztzcwd0vabw</sha1>
    </revision>
  </page>
  <page>
    <title>AmoeboidTaxa</title>
    <ns>0</ns>
    <id>24</id>
    <redirect title="Amoeba" />
    <revision>
      <id>627604809</id>
      <parentid>625443465</parentid>
      <timestamp>2014-09-29T22:26:03Z</timestamp>
      <contributor>
        <username>Invadibot</username>
        <id>15934865</id>
      </contributor>
      <minor />
      <comment>Bot: Fixing double redirect to [[Amoeba]]</comment>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text xml:space="preserve">#REDIRECT [[Amoeba]] {{R from CamelCase}}</text>
      <sha1>afkde9noo6ive9c3gr5pq9sqlqf6w64</sha1>
    </revision>
  </page>
  <page>
    <title>Autism</title>
    <ns>0</ns>
    <id>25</id>
    <revision>
      <id>692971972</id>
      <parentid>691895142</parentid>
      <timestamp>2015-11-29T16:00:39Z</timestamp>
      <contributor>
        <username>Iridescent</username>
        <id>937705</id>
      </contributor>
      <minor />
      <comment>/* Society and culture */[[WP:AWB/T|Typo fixing]], [[WP:AWB/T|typo(s) fixed]]: a influence → an influence, more more → more using [[Project:AWB|AWB]]</comment>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text xml:space="preserve">{{Hatnote|This article is about the classic autistic disorder; some writers use the word ''autism'' when referring to the range of disorders on the [[autism spectrum]] or to the various [[pervasive developmental disorder]]s.&lt;ref name=Caronna/&gt;}}
{{pp-semi-indef}}
{{pp-move-indef}}
{{bots|deny=Monkbot}} &lt;!-- keep Monkbot  from visiting this page --&gt;
{{Use dmy dates|date=June 2015}}
&lt;!-- NOTES:
1) Please follow the Wikipedia style guidelines for editing medical articles [[WP:MEDMOS]], and medical referencing standards at [[WP:MEDRS]].
2) Use &lt;ref&gt; for explicitly cited references.
3) Reference anything you put here with notable references, as this subject tends to attract a lot of controversy.--&gt;
{{Infobox disease
| Name = Autism
| Image = Autism-stacking-cans 2nd edit.jpg
| Alt = Young red-haired boy facing away from camera, stacking a seventh can atop a column of six food cans on the kitchen floor. An open pantry contains many more cans.
| Caption = Repetitively stacking or lining up objects is associated with autism.
| field = [[Psychiatry]]
| DiseasesDB = 1142
| ICD10 = {{ICD10|F|84|0|f|80}}
| ICD9 = {{ICD9|299.00}}
| OMIM = 209850
| MedlinePlus = 001526
| eMedicineSubj = med
| eMedicineTopic = 3202
| eMedicine_mult = {{eMedicine2|ped|180}}
| MeshID = D001321
| GeneReviewsNBK = NBK1442
| GeneReviewsName = Autism overview
}}
&lt;!-- Definition and symptoms --&gt;
'''Autism''' is a [[neurodevelopmental disorder]] characterized by impaired [[Interpersonal relationship|social interaction]], [[language acquisition|verbal]] and [[non-verbal communication]], and restricted and repetitive behavior. Parents usually notice signs in the first two years of their child's life.&lt;ref name=CCD/&gt; These signs often develop gradually, though some children with autism reach their [[developmental milestones]] at a normal pace and then [[Regressive autism|regress]].&lt;ref name=Stefanatos/&gt; The [[Diagnostic and Statistical Manual of Mental Disorders|diagnostic criteria]] require that symptoms become apparent in early childhood, typically before age three.&lt;ref name=DSM5&gt;{{vcite book | title = Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition | chapter = Autism Spectrum Disorder, 299.00 (F84.0) | editor = American Psychiatric Association | year = 2013 | publisher = American Psychiatric Publishing | pagex = 50–59}}&lt;/ref&gt;

&lt;!-- Causes and diagnosis --&gt;
While autism is highly heritable, researchers suspect both environmental and genetic factors as causes.&lt;ref&gt;{{cite journal |author=Chaste P, Leboyer M |title=Autism risk factors: genes, environment, and gene-environment interactions |journal=Dialogues in Clinical Neuroscience |volume=14  |pages=281–92 |year=2012 |pmid=23226953 |pmc=3513682 }}&lt;/ref&gt; In rare cases, autism is strongly associated with [[Teratology|agents that cause birth defects]].&lt;ref name=Arndt/&gt; [[Controversies in autism|Controversies]] surround other proposed environmental [[Causes of autism|causes]];&lt;ref name=Rutter/&gt; for example, the [[MMR vaccine controversy|vaccine hypotheses]] have been disproven. Autism affects information processing in the [[Human brain|brain]] by altering how [[nerve cell]]s and their [[synapse]]s connect and organize; how this occurs is not well understood.&lt;ref name=&quot;Levy&quot; /&gt; It is one of three recognized disorders in the [[autism spectrum]] (ASDs), the other two being [[Asperger syndrome]], which lacks delays in cognitive development and language, and [[PDD-NOS|pervasive developmental disorder, not otherwise specified]] (commonly abbreviated as PDD-NOS), which is diagnosed when the full set of criteria for autism or Asperger syndrome are not met.&lt;ref name=&quot;Johnson&quot; /&gt;

&lt;!-- Treatment --&gt;
Early speech or [[Early intensive behavioral intervention|behavioral interventions]] can help children with autism gain self-care, social, and communication skills.&lt;ref name=CCD/&gt; Although there is no known cure,&lt;ref name=CCD/&gt; there have been reported cases of children who recovered.&lt;ref name=Helt/&gt; Not many children with autism live independently after reaching adulthood, though some become successful.&lt;ref name=&quot;Howlin&quot;&gt;{{cite journal | vauthors = Howlin P, Goode S, Hutton J, Rutter M | title = Adult outcome for children with autism | journal = J Child Psychol Psychiatry | volume = 45 | issue = 2 | pages = 212–29 | year = 2004 | doi = 10.1111/j.1469-7610.2004.00215.x | pmid = 14982237}}&lt;/ref&gt; An [[Sociological and cultural aspects of autism|autistic culture]] has developed, with some individuals seeking a cure and others believing autism should be [[Autism rights movement|accepted as a difference and not treated as a disorder]].&lt;ref name=Silverman/&gt;

&lt;!-- Epidemiology --&gt;
Globally, autism is estimated to affect 21.7 million people as of 2013.&lt;ref name=&quot;Collab&quot;&gt;{{cite journal |author = Global Burden of Disease Study 2013 Collaborators |title=Global, regional, and national incidence, prevalence, and years lived with disability for 301 acute and chronic diseases and injuries in 188 countries, 1990–2013: a systematic analysis for the Global Burden of Disease Study 2013.|journal=Lancet|year = 2015|pmid=26063472 |doi=10.1016/S0140-6736(15)60692-4}}&lt;/ref&gt; As of 2010, the number of people affected is estimated at about 1–2 per 1,000 worldwide. It occurs four to five times more often in boys than girls. About 1.5% of children in the United States (one in 68) are diagnosed with ASD {{as of|2014|lc=y}}, a 30% increase from one in 88 in 2012.&lt;ref name=&quot;ASD Data and Statistics&quot;&gt;{{cite web |url = http://www.cdc.gov/ncbddd/autism/data.html |title = ASD Data and Statistics |website = CDC.gov |accessdate= 5 April 2014 |archiveurl = https://web.archive.org/web/20140418153648/http://www.cdc.gov/ncbddd/autism/data.html |archivedate = 18 April 2014 }}&lt;/ref&gt;&lt;ref name=&quot;MMWR2012&quot;&gt;{{cite journal | vauthors =  | title = Prevalence of autism spectrum disorders&amp;nbsp;— autism and developmental disabilities monitoring network, 14 sites, United States, 2008 | journal = MMWR Surveill Summ | volume = 61 | issue = 3 | pages = 1–19 | year = 2012 | pmid = 22456193 | url = http://www.cdc.gov/mmwr/preview/mmwrhtml/ss6103a1.htm | archivedate = 25 March 2014 | archiveurl = https://web.archive.org/web/20140325235639/http://www.cdc.gov/mmwr/preview/mmwrhtml/ss6103a1.htm }}&lt;/ref&gt;&lt;ref name=&quot;NHSR65&quot;&gt;{{cite journal | vauthors = Blumberg SJ, Bramlett MD, Kogan MD, Schieve LA, Jones JR, Lu MC | title = Changes in prevalence of parent-reported autism spectrum disorder in school-aged U.S. children: 2007 to 2011–2012 | journal = Natl Health Stat Report | volume = | issue = 65 | pages = 1–11 | year = 2013 | pmid = 24988818 | url = http://www.cdc.gov/nchs/data/nhsr/nhsr065.pdf | archiveurl = http://www.webcitation.org/6JoG0uE7r|archivedate=21 September 2013 }}&lt;/ref&gt; The rate of autism among adults aged 18 years and over in the United Kingdom is 1.1%.&lt;ref name=NHSEstimating/&gt; The number of people diagnosed has been increasing dramatically since the 1980s, partly due to changes in diagnostic practice and government-subsidized financial incentives for named diagnoses;&lt;ref name=&quot;NHSR65&quot;/&gt; the question of whether actual rates have increased is unresolved.&lt;ref name=Newschaffer/&gt;

==Characteristics==
Autism is a highly variable [[neurodevelopmental disorder]]&lt;ref name=Geschwind/&gt; that first appears during infancy or childhood, and generally follows a steady course without [[Remission (medicine)|remission]].&lt;ref name=ICD-10-F84.0/&gt; People with autism may be severely impaired in some respects but normal, or even superior, in others.&lt;ref&gt;{{vcite book|title = Biopsychology|author= Pinel JPG|publisher = Pearson|year = 2011|isbn = 978-0-205-03099-6|location = Boston, Massachusetts|page = 235}}&lt;/ref&gt; Overt symptoms gradually begin after the age of six months, become established by age two or three years,&lt;ref&gt;{{cite journal | vauthors = Rogers SJ | title = What are infant siblings teaching us about autism in infancy? | journal = Autism Res | volume = 2 | issue = 3 | pages = 125–37 | year = 2009 | pmid = 19582867 | pmc = 2791538 | doi = 10.1002/aur.81  }}&lt;/ref&gt; and tend to continue through adulthood, although often in more muted form.&lt;ref name=Rapin/&gt; It is distinguished not by a single symptom, but by a characteristic triad of symptoms: impairments in social interaction; impairments in communication; and restricted interests and repetitive behavior. Other aspects, such as atypical eating, are also common but are not essential for diagnosis.&lt;ref name=Filipek/&gt; Autism's individual symptoms occur in the general population and appear not to associate highly, without a sharp line separating pathologically severe from common traits.&lt;ref name=London/&gt;

==References==
{{reflist|32em}}

==Further reading==
*{{vcite book|author=Sicile-Kira, C|title=Autism spectrum disorder: the complete guide to understanding autism|date=2014|publisher=Perigee|location=New York, New York|isbn=978-0-399-16663-1|edition=Revised Perigee trade paperback}}
*{{vcite book|author=Waltz, M|title=Autism: A Social and Medical History|date=22 March 2013|publisher=Palgrave Macmillan|isbn=978-0-230-52750-8|edition=1st}}
*{{vcite book|author=[[Steve Silberman|Silberman, S]]|title=NeuroTribes: The Legacy of Autism and How to Think Smarter About People Who Think Differently|date=2015|publisher=[[Allen &amp; Unwin]]|location=Crows Nest, New South Wales|isbn=978-1-760-11363-6|edition=1st}}

==External links==
{{Sister project links|d=Q38404|s=no|n=Category:Autism|wikt=autism|species=no|voy=no|m=no|mw=no|q=no|v=no}}
* {{dmoz|Health/Mental_Health/Disorders/Neurodevelopmental/Autism_Spectrum}}

{{featured article}}
{{Pervasive developmental disorders}}
{{Mental and behavioral disorders|selected = childhood}}
{{Autism resources}}
{{Autism films}}
{{Portal bar|Pervasive developmental disorders}}

[[Category:Autism| ]]
[[Category:Communication disorders]]
[[Category:Mental and behavioural disorders]]
[[Category:Neurological disorders]]
[[Category:Neurological disorders in children]]
[[Category:Pervasive developmental disorders]]
[[Category:Psychiatric diagnosis]]</text>
      <sha1>jrwedqcbxp53m5c4g7umfqmh2alo26n</sha1>
    </revision>
  </page>
</mediawiki>
""".encode('ascii', errors='ignore')  # ugh, Python 2 requires this


class WikiReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_corpora', dir=os.path.dirname(os.path.abspath(__file__)))
        wiki_fname = os.path.join(self.tempdir, 'wikitext.xml.bz2')
        write_file(WIKITEXT, wiki_fname, mode='wb', auto_make_dirs=True)
        self.wikireader = WikiReader(wiki_fname)

    def test_texts(self):
        texts = list(self.wikireader.texts())
        for text in texts:
            self.assertIsInstance(text, unicode_type)

    def test_texts_min_len(self):
        texts = list(self.wikireader.texts(min_len=300))
        self.assertEqual(len(texts), 1)

    def test_texts_limit(self):
        texts = list(self.wikireader.texts(limit=1))
        self.assertEqual(len(texts), 1)

    def test_records(self):
        records = list(self.wikireader.records())
        for record in records:
            self.assertIsInstance(record, dict)

    def test_records_min_len(self):
        records = list(self.wikireader.records(min_len=300))
        self.assertEqual(len(records), 1)

    def test_records_limit(self):
        records = list(self.wikireader.records(limit=1))
        self.assertEqual(len(records), 1)

    def tearDown(self):
        shutil.rmtree(self.tempdir)
