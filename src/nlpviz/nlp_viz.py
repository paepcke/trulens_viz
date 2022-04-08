'''
Created on Apr 5, 2022

@author: paepcke
'''

import tempfile
import time
from enum import Enum
import webbrowser

import numpy as np
import matplotlib

import domonic as dm

class WordStyles(Enum):
    FONT_SIZE  = 0
    FONT_COLOR = 1

# --------------- HTMLTable ---------------
class HTMLTable:
    '''
    Given list of (word, attribution_score) tuples,
    create and render an HTML table of two rows.
    The first row will contain the words, colored
    to reflect their attribution score. The second
    will contain the attribution scores.
    
    Uses a divergent color map, showing negative scores
    in shades of red, and positive scores in shades of 
    green.
    
    Usage:
        doc = HTMLTable(word_attributions)
        doc.render_to_web() 
    '''
    # When using font size or color to reflect word scores:
    # number of quantile bins for font size/color gradations
    NUM_BINS  = 5

    # Color map when reflecting word scores using color
    #cmap_name = 'PiYG'
    cmap_name = 'YlGn'
    # Percentage of the chosen color map's width
    # for each bin. Percentage 0.5 corresponds to the
    # middle of the colormap. Choose to contrast with 
    # background:
    FONT_COLOR_LOOKUP = {
       0 : 0.4,
       1 : 0.5,
       2 : 0.6,
       3 : 0.7,
       4 : 1.
       }

    # Color bin below which the text is light
    # enough that the background of the text
    # (i.e. the table cell color) should be darkened 
    # for visibility. Set to None for no background darkening:
    DARKEN_BACKGROUND_THRES = 3
    DARK_BACKGROUND = 'Gray'
    # Lookup for font size in percent of <body> font.
    # There need to be as many entries in this dict
    # as there are bins:
    FONT_SIZE_LOOKUP = {0: 100,
                        1: 250,
                        2: 400,
                        3: 600,
                        4: 1300
                        }

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, word_attributions, word_styling=WordStyles.FONT_SIZE):
        '''
        Constructs a domonic HTML document. The
        instance will be ready for client invoking
        its render_to_web() method without arguments. 
        
        :param word_attributions: list of tuples, each containing a word and its score
        :type word_attributions: [(str, float)]
        '''

        self.all_word_attributions = np.array([])
        self.row_word_styles = {}

        self.add_rows(word_attributions, word_styling)
        
    #------------------------------------
    # add_rows
    #-------------------
    
    def add_rows(self, word_attributions, word_styling=WordStyles.FONT_SIZE):
        '''
        Takes an array of word/score pair arrays. Each array
        contains all word/score pairs of one phrase.
            [[('The', 142.0), ('man', -13.) ('lies', 410.0)]
             [('My' -143.5), ('jacket', 53.0), ('is' 111.45), ('very' -1887), ('blue', 10.)]
             ]
        A human-presentable representation is created for
        earch phrase. The phrase representations are added
        to the ones already in self.all_word_attributions. Then
        an HTML table is constructed with all phrases.
         
        :param word_attributions:
        :type word_attributions:
        :param word_styling:
        :type word_styling:
        '''
        
        if type(word_styling) != WordStyles:
            raise ValueError(f'Bad word style: {word_styling}')
        
        if type(word_attributions) != np.ndarray:
            word_attrs_np = np.array(word_attributions)
            # If only one phrase was passed in, adjust
            # the np array to have an additional axis that
            # would hold multiple sentences, but in this
            # case will only have one element:
            if word_attrs_np.ndim < 3:
                # the 2 are the word/score pairs:
                word_attrs_np = word_attrs_np.reshape(1, -1, 2)

        # For each phrase, clean its words so as not to
        # conflict with HTML conventions. Add the phrase
        # to self.all_word_attributions:
        
        for phrase in word_attrs_np:
            phrase_word_attrs = []
            for word_score in phrase:
                phrase_word_attrs.append(self.canonicalize_word_attr(word_score))
            phrase_word_attrs_np = np.array(phrase_word_attrs).reshape(1,-1,2) 
            # If no attrs have been processed, this one is our first:
            if self.all_word_attributions.ndim == 1:
                self.all_word_attributions = phrase_word_attrs_np
            else:
                width_adjusted_word_attrs = self.adjust_table_width(phrase_word_attrs_np)
                self.all_word_attributions = np.vstack((self.all_word_attributions, 
                                                        width_adjusted_word_attrs))
            # Note how this row's words are to be styled:
            self.row_word_styles[len(self.all_word_attributions) - 1] = word_styling

        # Create a new tbl instance, and update the word--bin_id
        # lookup dict:
        self.doc = self.prep_table()
        
        # Create a row-pair for each phrase (styled words in first row),
        # and scores in second row):
        for row_num, phrase in enumerate(self.all_word_attributions):
            if self.row_word_styles[row_num] == WordStyles.FONT_COLOR: 
                styled_words = self.create_colored_words(phrase)
            elif self.row_word_styles[row_num] == WordStyles.FONT_SIZE:
                styled_words = self.create_font_sized_words(phrase)
                
                
            html_words_row  = self.tbl.appendChild(dm.HTMLTableRowElement())
            html_scores_row = self.tbl.appendChild(dm.HTMLTableRowElement())
    
            for i, styled_word in enumerate(styled_words):
                if styled_word.darken_background:
                    tbl_cell_style = f'background-color : {self.DARK_BACKGROUND}'
                else:
                    tbl_cell_style = ''
                html_words_row.appendChild(dm.HTMLTableCellElement(styled_word,
                                                                   style=tbl_cell_style))
                attr_score = round(float(phrase[i,1]),2)
                html_scores_row.appendChild(dm.HTMLTableCellElement(attr_score))

    #------------------------------------
    # prep_table
    #-------------------
    
    def prep_table(self):
        
        # Get a list of all scores, across all phrases:
        # np array of all_word_attributions is of shape (1, num_phrases, 2),
        # where the 2-dimension holds the (word, attr_score). The squeeze
        # removes the outer dim:
        all_scores = self.all_word_attributions[:,:,1].astype(float)
        all_words  = self.all_word_attributions[:,:,0]
        self.bin_lookup = {}
        for words_1phrase, bin_ids_1phrase in zip(all_words, 
                                                  QuantileBinner.qcut(all_scores, 
                                                                      self.NUM_BINS)):
            self.bin_lookup.update({word : bin_id 
                                    for word, bin_id 
                                    in zip(words_1phrase, bin_ids_1phrase)})

        doc = dm.html(dm.head(), dm.body())
        style = self.create_style()
        doc.head.appendChild(style)
        self.tbl   = self.create_table_skeleton()
        doc.body.appendChild(self.tbl)
        
        return doc

    #------------------------------------
    # make_style
    #-------------------
    
    def create_style(self):
        '''
        Create and return a domonic HTMLStyleElement,
        suitable for appending the a document's HEAD.
        
        :returns a style element containing all styles
            required by the document
        :rtype: dm.HTMLStyleElement
        '''
        
        style = dm.HTMLStyleElement('''
              table, th, td {border: 1px solid;
                             border-collapse: collapse;
                            }
              td {text-align:center;
                  padding:10px;
                 }
              tr:nth-child(odd) {background-color: DarkGray;}
        ''')
        return style

    #------------------------------------
    # create_table_skeleton
    #-------------------
    
    def create_table_skeleton(self):
        '''
        Returns a domonic HTMLTableElement, ready for the
        addition of row elements via add_row()

        :return: a table HTML element
        :rtype: dm.HTMLTableElement

        '''
        tbl = dm.HTMLTableElement()
        return tbl

    #------------------------------------
    # create_font_sized_words
    #-------------------
    
    def create_font_sized_words(self, word_attr_scores):
        '''
        Return a list of domonic HTMLSpanElement. Each element of 
        the list will be a <span> that styles the word's font size
        to reflect its score.
        
        The arg is a list of (word, score) tuples that correspond
        to one phrase. In addition, each span element will contain
        a boolean attribute darken_background, which is set to False. 
        This attr is used by the word color method only. 

        Each HTMLSpanElement is ready to insert into a domonic html 
        table cell.
        
        :param word_attr_scores: list of (word, attributionScore) tuples
        :type word_attr_scores: [(str, float)]
        :returns list of html <span> snippets
        :type [dm.HTMLSpanElement]
        '''

        # Map each word's logit score into one of NUM_BIN bins
        output = []
        for word, _attr_score in word_attr_scores: 
            bin_id = self.bin_lookup[word]
            # Each font step sizes fonts by 100%. I.e. 
            # smallest is 100% of body font. Next size
            # is 200% of body font, etc.
            #*******font_perc = 100 + bin_id * 100
            #*******font_perc = 100 + np.e**(bin_id/2) * 100
            #*******font_perc = 100 + bin_id * 350
            #*******font_perc = 100 + np.sqrt(3*bin_id)
            #font_perc = 100 + 6**bin_id
            font_perc = self.FONT_SIZE_LOOKUP[bin_id]
            word_style = f'font-size:{font_perc}%;'
            span_el = dm.HTMLSpanElement(word, style=word_style)
            # No special table cell background coloring for thise
            # word:
            span_el.darken_background = False 
            output.append(span_el)
  
        return output

    #------------------------------------
    # create_colored_words
    #-------------------
    
    def create_colored_words(self, word_attr_scores):
        '''
        Return a list of domonic HTMLSpanElement. Each element of 
        the list will be a <span> that styles the word's color 
        to reflect its score.
        
        The arg is a list of (word, score) tuples that correspond
        to one phrase. In addition, each span element will contain
        a boolean attribute darken_background. If true, the word
        color is light enough that the caller should darken the
        background on which the word will appear.

        Each HTMLSpanElement is ready to insert into a domonic html 
        table cell.
        
        :param word_attr_scores: list of (word, attributionScore) tuples
        :type word_attr_scores: [(str, float)]
        :returns list of html <span> snippets
        :type [dm.HTMLSpanElement]
        '''

        # Pick a colormap; see https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html:
        cmap = matplotlib.cm.get_cmap(self.cmap_name)
        
        output = []
        for word, _attr_score in word_attr_scores: 
            bin_id = self.bin_lookup[word]
            color  = (np.array(cmap(self.FONT_COLOR_LOOKUP[bin_id])) * 255).astype(int)
            word_style = f'color:rgb{tuple(color)}; font-size:200%; font-weight:bold;'
            span_el = dm.HTMLSpanElement(word, style=word_style)
            # Is the color light enough that the background
            # of the text should be darkened for visibility?
            if self.DARKEN_BACKGROUND_THRES is not None and bin_id < self.DARKEN_BACKGROUND_THRES:
                span_el.darken_background = True
            else:
                span_el.darken_background = False
            output.append(span_el)

        # Start REMOVE
        # # Map logits to [0,1]:
        # normed_mags = matplotlib.colors.Normalize(vmin=min(attr_scores), 
        #                                           vmax=max(attr_scores))(attr_scores)
        #
        # # Pick colors from the colormap, proportional to logit value:
        # colors = []
        # for mag in normed_mags:
        #     # Pick color from colormap, dropping the
        #     # (always 255) opacity value:
        #     colors.append(cmap(mag, bytes=True)[:3])
        #
        # # Get the whole phrase as HTML, escaping occurrences
        # # of '<':
        # # output = [f"<span title='{mag:0.3f}' style='margin: 1px; padding: 1px; border-radius: 4px; background: black; color: rgb{color};'>{word.replace('<', '&lt')}</span>"
        # #           for word, color, mag
        # #           in zip(words, colors, attr_mags)]
        # # output = [f"<span title='{mag:0.3f}' style='color: rgb{color};'>{word.replace('<', '&lt')}</span>"
        # #           for word, color, mag
        # #           in zip(words, colors, attr_mags)]
        # output = [dm.HTMLSpanElement(word.replace('<', '&lt'),
        #             style=dm.HTMLStyleElement(f'id:{mag:0.3f}; color:rgb{color};'))
        #           for word, color, mag
        #           in zip(words, colors, attr_scores)]
        # END REMOVE
        return output

    #------------------------------------
    # render_to_web
    #-------------------
    
    def render_to_web(self):
        '''
        Given an HTML string, open the default browser, and
        display the string there.
        
        The name of a temp file is returned. It will be be
        deleted automatically upon garbage collection.
    
        :param output: HTML to display
        :type output: str
        :return temp file path
        :rtype str
        '''
        fd = tempfile.NamedTemporaryFile(prefix='attrs_', suffix='.html')
        fd.write(bytes('<html>', 'utf8'))
        fd.write(bytes(str(self.doc), 'utf8'))
        fd.write(bytes('</html>', 'utf8'))
        fd.flush()
        webbrowser.open_new_tab(f"file://{fd.name}")
        # Terrible hack! Must ensure the browser has read
        # the file before removing the temp file. The right
        # way would be to check for presence of some page
        # element:
        time.sleep(5)
        return fd.name
    
    #------------------------------------
    # canonicalize_word_attr
    #-------------------
    
    def canonicalize_word_attr(self, word_attribution):
        '''
        Given one word/score pair, modify the word or
        attribution score for use with HTML.
        
        :param word_attribution: word/attribution_score to clean
        :type word_attribution: (str, {str | float | int})
        :return an HTML-safe tuple
        :rtype (str, float)
        '''
        word, score = word_attribution
        # Replace HTML tag opener token:
        return (word.replace('<', '&lt'), float(score))

    #------------------------------------
    # adjust_table_width
    #-------------------
    
    def adjust_table_width(self, new_phrase_data):
        
        _num_phrases, phrase_width, _tuple_len = new_phrase_data.shape
        _num_all_phrases, all_phrase_width, _tuple_len = self.all_word_attributions.shape
        
        if phrase_width == all_phrase_width:
            return new_phrase_data
        
        if all_phrase_width < phrase_width:
            # Widen the already existing table:
            for _i in range(phrase_width - all_phrase_width):
                self.all_word_attributions = np.hstack((self.all_word_attributions, 
                                                        [[('','0')]]))
        else:
            # Widen the new phrase:
            for _i in range(all_phrase_width - phrase_width):
                new_phrase_data = np.hstack((new_phrase_data,
                                                  [[('','0')]]))
        return new_phrase_data
        

# ------------------- Class Binner ------------   

class Binner:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, in_range, out_range, num_bins):
        
        self.in_min, self.in_max   = in_range
        self.out_min, self.out_max = out_range
        self.create_bins(num_bins)

    #------------------------------------
    # select_bin
    #-------------------

    def select_bin(self, in_value):
        
        val = self.map_range(in_value)
        for bin_id, _bin in enumerate(self.bins):
            bin_low, bin_high = _bin
            if val >= bin_low and val < bin_high:
                return bin_id
        return -1

    #------------------------------------
    # map_range
    #-------------------
     
    def map_range(self, value):
        if value >= self.out_min and value < self.out_max:
            return value
        if value < self.in_min or value > self.in_max:
            raise IndexError(f"Value {value} not in [{self.bins[0][0]}, {self.bins[-1][1]})")
        return self.out_min + (((value - self.in_min) / (self.in_max - self.in_min)) * (self.out_max - self.out_min))

    #------------------------------------
    # create_bins
    #-------------------
    
    def create_bins(self, num_bins):

        bins = []
        width = (self.out_max - self.out_min) / num_bins 
        
        low = self.out_min
        while low < self.out_max:
            bins.append((low, low+width))
            low += width
        # Make the last bin inclusive of the max 
        # val of out-range. I.e. make the max out 
        # range value satisfy "< last_box_high":
        bins[-1] = (bins[-1][0], bins[-1][1] + 0.0001)
        self.bins = bins

# ------------------- Class QuantileBinner ----------

class QuantileBinner:
    
    @staticmethod
    def qcut(x, bin_info):
        """
        Quantile-based discretization function.
        Discretize variable into equal-sized buckets based on sample quantiles. 
        For example 1000 values for 10 quantiles would
        produce a Categorical object indicating quantile membership for each data point.
        
        Returns
        -------
        out : Categorical or Series or array of integers if labels is False
            The return type (Categorical or Series) depends on the input: a Series
            of type category if input is a Series else Categorical. Bins are
            represented as categories when categorical data is returned.
        bins : ndarray of floats
            Returned only if `retbins` is True.
        Notes
        -----
        Out of bounds values will be NA in the resulting Categorical object
        Examples
        --------
        >>> pd.qcut(range(5), 4)
        ... # doctest: +ELLIPSIS
        [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
        Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] ...
        >>> pd.qcut(range(5), 3, labels=["good", "medium", "bad"])
        ... # doctest: +SKIP
        [good, good, medium, bad, bad]
        Categories (3, object): [good < medium < bad]
        >>> pd.qcut(range(5), 4, labels=False)
        array([0, 0, 1, 2, 3])
        """
        x_np = np.asarray(x)
        x_np.sort()
        x_np = np.unique(x_np)
        x_np = x_np[~np.isnan(x_np)]
        
        # the +1 replaces the zeroeth percentile, which is
        # removed by the [1:]:
        quantiles = np.linspace(0, 1, bin_info + 1)[1:] if type(bin_info) == int else bin_info
        
        bins = np.quantile(x_np, quantiles)
    
        ids = QuantileBinner._bins_to_cuts(x, bins)
    
        return ids
    
    @staticmethod
    def _bins_to_cuts(x, unique_bins):
        '''
        
        :param x: sorted array of numbers that are to be assigned
            into bins
        :type x: np.array
        :param unique_bins: either a single int generate quantiles,
            or a list of quantiles
        :type unique_bins: {int | [float]}
        :return
        :rtype
        '''
    
        ids = unique_bins.searchsorted(x)
        return ids
