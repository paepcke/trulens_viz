'''
Created on Apr 5, 2022

@author: paepcke
'''
import unittest
import numpy as np

from nlp_viz import Binner, HTMLTable, WordStyles, QuantileBinner


TEST_ALL = True
#TEST_ALL = False

class NLPVizTester(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    # ------------------ Tests Binning ----------------



    #------------------------------------
    # test_map_range
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_map_range(self):
        binner = Binner((-10, 10), (-1, 1), 5)
        with self.assertRaises(IndexError):
            binner.map_range(-11)
        self.assertEqual(binner.map_range(-10), -1)
        self.assertEqual(binner.map_range(10), 1)

    #------------------------------------
    # test_binner
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_binner(self):
        binner = Binner((-10, 10), (-1, 1), 5)
        self.assertEqual(len(binner.bins), 5)
        self.assertEqual(binner.select_bin(-10), 0)
        self.assertEqual(binner.select_bin(-.5), 1)
        self.assertEqual(binner.select_bin(10), 4)
        with self.assertRaises(IndexError):
            binner.select_bin(11)
        with self.assertRaises(IndexError):
            binner.select_bin(-10.1)
    
    # -------------------Tests for QuantileBinner ------------
    
    #------------------------------------
    # test_quantile_binner
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_quantile_binner(self):
        
        x = [-10345, -3, 6, -5, 6, 140]
        bin_info = 5 
        bin_ids  = QuantileBinner.qcut(x, bin_info)
        expected = np.array([0, 2, 3, 1, 3, 4])
        self.assertTrue((bin_ids == expected).all())

    # ------------------ Tests HTML Table Creation ----------------

    #------------------------------------
    # test_tbl_creation
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_tbl_creation(self):
        
        word_attrs = [('foo', -10345), ('<s>', -3), ('bar', 6)]
        tbl = HTMLTable(word_attrs, word_styling=WordStyles.FONT_SIZE)
        
        expected = '<html><head><style>\n              table, th, td {border: 1px solid;\n                             border-collapse: collapse;\n                            }\n              td {text-align:center;\n                  padding:10px;\n                 }\n              tr:nth-child(odd) {background-color: DarkGray;}\n        </style></head><body><table><tr><td style=""><span style="font-size:100%;">foo</span></td><td style=""><span style="font-size:400%;">&lts></span></td><td style=""><span style="font-size:1300%;">bar</span></td></tr><tr><td>-10345.0</td><td>-3.0</td><td>6.0</td></tr></table></body></html>'
        self.assertEqual(str(tbl.doc), expected)

    #------------------------------------
    # test_tbl_add_rows_constant_phrase_len
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_tbl_add_rows_constant_phrase_len(self):
        
        word_attrs = [('foo', -10345), ('<s>', -3), ('bar', 6)]
        tbl = HTMLTable(word_attrs, word_styling=WordStyles.FONT_SIZE)
        expected = '<html><head><style>\n              table, th, td {border: 1px solid;\n                             border-collapse: collapse;\n                            }\n              td {text-align:center;\n                  padding:10px;\n                 }\n              tr:nth-child(odd) {background-color: DarkGray;}\n        </style></head><body><table><tr><td style=""><span style="font-size:100%;">foo</span></td><td style=""><span style="font-size:400%;">&lts></span></td><td style=""><span style="font-size:1300%;">bar</span></td></tr><tr><td>-10345.0</td><td>-3.0</td><td>6.0</td></tr></table></body></html>'
        self.assertEqual(str(tbl.doc), expected)

        new_word_attrs = [('bluebell', -5), ('is', 6), ('pretty', 140)]
        tbl.add_rows(new_word_attrs, word_styling=WordStyles.FONT_SIZE)
        expected = '<html><head><style>\n              table, th, td {border: 1px solid;\n                             border-collapse: collapse;\n                            }\n              td {text-align:center;\n                  padding:10px;\n                 }\n              tr:nth-child(odd) {background-color: DarkGray;}\n        </style></head><body><table><tr><td style=""><span style="font-size:100%;">foo</span></td><td style=""><span style="font-size:400%;">&lts></span></td><td style=""><span style="font-size:600%;">bar</span></td></tr><tr><td>-10345.0</td><td>-3.0</td><td>6.0</td></tr><tr><td style=""><span style="font-size:250%;">bluebell</span></td><td style=""><span style="font-size:600%;">is</span></td><td style=""><span style="font-size:1300%;">pretty</span></td></tr><tr><td>-5.0</td><td>6.0</td><td>140.0</td></tr></table></body></html>'
        self.assertEqual(str(tbl.doc), expected)

    #------------------------------------
    # test_tbl_add_rows_varied_phrase_len
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_tbl_add_rows_varied_phrase_len(self):
        
        word_attrs = [('foo', -10345), ('<s>', -3), ('bar', 6)]
        tbl = HTMLTable(word_attrs, word_styling=WordStyles.FONT_SIZE)
        expected = '<html><head><style>\n              table, th, td {border: 1px solid;\n                             border-collapse: collapse;\n                            }\n              td {text-align:center;\n                  padding:10px;\n                 }\n              tr:nth-child(odd) {background-color: DarkGray;}\n        </style></head><body><table><tr><td style=""><span style="font-size:100%;">foo</span></td><td style=""><span style="font-size:400%;">&lts></span></td><td style=""><span style="font-size:1300%;">bar</span></td></tr><tr><td>-10345.0</td><td>-3.0</td><td>6.0</td></tr></table></body></html>'
        self.assertEqual(str(tbl.doc), expected)

        new_word_attrs = [('bluebell', -5), ('is', 6), ('pretty', 140), ('grand', 10)]
        tbl.add_rows(new_word_attrs, word_styling=WordStyles.FONT_SIZE)
        expected = '<html><head><style>\n              table, th, td {border: 1px solid;\n                             border-collapse: collapse;\n                            }\n              td {text-align:center;\n                  padding:10px;\n                 }\n              tr:nth-child(odd) {background-color: DarkGray;}\n        </style></head><body><table><tr><td style=""><span style="font-size:100%;">foo</span></td><td style=""><span style="font-size:250%;">&lts></span></td><td style=""><span style="font-size:400%;">bar</span></td><td style=""><span style="font-size:600%;"></span></td></tr><tr><td>-10345.0</td><td>-3.0</td><td>6.0</td><td>0.0</td></tr><tr><td style=""><span style="font-size:100%;">bluebell</span></td><td style=""><span style="font-size:600%;">is</span></td><td style=""><span style="font-size:1300%;">pretty</span></td><td style=""><span style="font-size:1300%;">grand</span></td></tr><tr><td>-5.0</td><td>6.0</td><td>140.0</td><td>10.0</td></tr></table></body></html>'
        self.assertEqual(str(tbl.doc), expected)

    #------------------------------------
    # test_color_viz
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_color_viz(self):
        word_attrs = [('foo', -10345), ('<s>', -3), ('bar', 6)]
        tbl = HTMLTable(word_attrs, word_styling=WordStyles.FONT_COLOR)
        #print(str(tbl.doc))

        tbl.add_rows([('My', -12345.), 
                      ('Bonny', -100),
                       ('lies', 0), 
                       ('over', 100), 
                       ('the', 500), 
                       ('ocean', 10000.)],
                      word_styling=WordStyles.FONT_SIZE)
        
        tbl.add_rows([('My', -12345.), 
                      ('Bonny', -100),
                       ('lies', 0), 
                       ('over', 100), 
                       ('the', 500), 
                       ('ocean', 10000.)],
                      word_styling=WordStyles.FONT_COLOR)
        
        print(str(tbl.doc))
        print(tbl)

    #------------------------------------
    # test_adjust_table_width
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_adjust_table_width(self):
        
        word_attrs = [('foo', -10345), ('<s>', -3), ('bar', 6)]
        tbl = HTMLTable(word_attrs, word_styling=WordStyles.FONT_SIZE)

        # Now tbl.all_word_attributions is shape (1,3,2)

        # Make a new phrase by copying the one above: i.e. same shape:
        new_phrase_data = np.array(word_attrs).reshape(1,-1,2)
        width_adjusted = tbl.adjust_table_width(new_phrase_data)
        # Since the new phrase has same width as old one, nothing 
        # should have been changed:
        self.assertTrue((width_adjusted == new_phrase_data).all())

        # New phrase, wider by one word than the already added row:
        wide_word_attrs = np.array([('bluebell', -5), 
                                    ('is', 6), 
                                    ('pretty', 140), 
                                    ('grand', 10)]).reshape(1,-1,2)
        self.assertEqual(wide_word_attrs.shape, (1,4,2))

        # Remember the current shape of the table, since
        # it should change:
        prior_all_tbl   = tbl.all_word_attributions.copy()
        self.assertEqual(prior_all_tbl.shape, (1,3,2))
        
        # Widen the already existing table to match to new 
        # wider phrase on all rows:
        width_adjusted = tbl.adjust_table_width(wide_word_attrs)
        
        self.assertEqual(tbl.all_word_attributions.shape, (1,4,2))
        self.assertEqual(wide_word_attrs.shape, (1,4,2))
        
        # Other way aroud: new phrase narrower than current table:
        wide_word_attrs = np.array([('Gray', -10), 
                                    ('ocean', 30)]).reshape(1,-1,2)
        prior_all_tbl   = tbl.all_word_attributions.copy()
        self.assertEqual(prior_all_tbl.shape, (1,4,2))
        self.assertEqual(wide_word_attrs.shape, (1,2,2))
        
        # Widen the new phrase:
        width_adjusted = tbl.adjust_table_width(wide_word_attrs)
        
        self.assertEqual(tbl.all_word_attributions.shape, (1,4,2))
        self.assertEqual(width_adjusted.shape, (1,4,2))


# ---------------- Main ------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()