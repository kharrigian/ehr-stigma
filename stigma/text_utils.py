
####################
### Imports
####################

## Standard Library
import re
import sys
import string
import warnings
from collections import Counter

## External
import spacy
import numpy as np
from tqdm import tqdm
from scipy import sparse
from pysbd.utils import PySBDFactory
from medspacy.custom_tokenizer import create_medspacy_tokenizer
from unidecode import unidecode

######################
### Globals
######################

## Useful Regular Expressions
_WS_REP = re.compile(r"\s+", re.UNICODE|re.IGNORECASE)

######################
### Functions
######################

def get_vocabulary(tokens,
                   rm_top=None,
                   min_freq=None,
                   max_freq=None,
                   stopwords=set()):
    """
    
    """
    ## Get Count of Unique Tokens
    counts = Counter()
    for t in tokens:
        counts.update(t)
    ## Filter
    counts = counts.most_common()
    if rm_top:
        counts = counts[rm_top:]
    if min_freq:
        counts = [(x,y) for x, y in counts if y >= min_freq]
    if max_freq:
        counts = [(x,y) for x, y in counts if y <= max_freq]
    if stopwords:
        counts = [(x,y) for x, y in counts if x not in stopwords]
    ## Isolate Vocabulary
    vocabulary = [c[0] for c in counts]
    ## Mapping
    vocabulary2ind = dict(zip(vocabulary, range(len(vocabulary))))
    return vocabulary, vocabulary2ind

def get_ngrams(tokens, n):
    """
    
    """
    ## Get N-Grams
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return ngrams

def clean_excel_text(text):
    """
    Extracted from Ayah's NoteCleaner.py. Note that the unicodedate
    functions are very slow and have been replaced by Keith with unidecode
    """
    ## Check Type
    if not isinstance(text, str):
        return text
    ## Clean
    clean = text
    clean = re.sub('[\t\r\n]', '   ', clean)
    clean = re.sub(u'\xa0 ', ' ', clean)
    clean = re.sub("'", '"', clean)
    clean = re.sub('&gt;', '>', clean)
    clean = re.sub('&lt;', '<', clean)
    clean = re.sub('&amp;', ' and ', clean)
    clean = re.sub('&#x20;', ' ', clean)
    clean = re.sub('   ', '\n', clean)
    clean = re.sub('\u2022', '\t', clean)
    clean = re.sub('\x1d|\xa0|\x1c|\x14', ' ', clean)
    clean = re.sub('### invalid font number [0-9]+', ' ', clean)
    clean = re.sub('[ ]+', ' ', clean)
    clean = unidecode(clean)
    return clean

def _strip_operators(text):
    """

    """
    if not isinstance(text, str):
        return text
    text = text.lstrip("-+*/=")
    text = text.strip()
    return text

def normalize_excel_text(text,
                         strip_operators=False):
    """
    Applies:
        - lowercase
        - reduction of multiple whitespaces into one
        - strip leading and trailing whitespace
    """
    ## Check Type
    if not isinstance(text, str):
        return None
    ## Format
    text = text.lower()
    text = _WS_REP.sub(" ", text)
    text = text.strip()
    if strip_operators:
        text = _strip_operators(text)
    return text

def get_context_window(text,
                       start,
                       end,
                       window_size=10,
                       clean_normalize=True,
                       strip_operators=False):
    """
    
    """
    ## Update White-space Limits
    while start > 0 and text[start - 1] != " ":
        start -= 1
    while end < len(text) - 1 and text[end] != " ":
        end += 1
    ## Get Left/Right/Center
    text_left = text[:start].strip().split(" ")[-window_size:]
    text_span = text[start:end].strip().split(" ")
    text_right = text[end:].strip().split(" ")[:window_size]
    ## Merge
    text_window = " ".join(text_left + text_span + text_right)
    ## Clean/Normalize
    if clean_normalize:
        text_window = clean_excel_text(text_window)
        text_window = normalize_excel_text(text_window, strip_operators=False)
    ## Strip Operators
    if strip_operators:
        text_window = _strip_operators(text_window)
    return text_window

def extract_negated_keyword(keyword, text):
    """
    
    """
    match = re.search(r"(cannot|not|\w*n't)(\s)?\b{}\b".format(keyword), text, re.IGNORECASE)
    if match is None:
        return keyword
    return text[match.start():match.end()]

def _get_medical_tokenizer_old():
    """
    For spacy version 2 (used on the MCEH-Bridge Machine)
    """
    ## Initialize Blank Class
    nlp = spacy.blank("en")
    ## Create Tokenizer and Add to Class
    nlp.tokenizer = create_medspacy_tokenizer(nlp)
    ## Add Sentence Segmentation
    try:
        nlp.add_pipe(PySBDFactory(nlp),first=True)
    except:
        nlp.add_pipe(nlp.create_pipe('sentencizer'),first=True)
        warnings.warn("WARNING: Using sentencizer instead of medical sentence segmentation.")
    return nlp

def _get_medical_tokenizer():
    """
    For spacy version 3. use get_medical_tokenizer_old for spacy version 2.
    """
    ## Initialize Blank Class
    nlp = spacy.blank("en")
    ## Create Tokenizer and Add to Class
    nlp.tokenizer = create_medspacy_tokenizer(nlp)
    ## Add Sentence Segmentation
    try:
        nlp.add_pipe("medspacy_pysbd",first=True)
    except:
        nlp.add_pipe("sentencizer",first=True)
        warnings.warn("WARNING: Using sentencizer instead of medical sentence segmentation.")
    return nlp

def get_medical_tokenizer():
    """
    
    """
    try:
        return _get_medical_tokenizer()
    except:
        return _get_medical_tokenizer_old()

######################
### Classes
######################

class KeywordSearch(object):

    """
    
    """

    def __init__(self,
                 task2keyword):
        """
        Args:
            task2keyword (dict): {"task 1":["keyword 1","keyword 2",...], "task 2":...}
        """
        self._task2keyword = task2keyword
        self._task2keyword_re = self._initialize_search_patterns(task2keyword)
    
    def __repr__(self):
        """
        
        """
        return "KeywordSearch()"

    def _initialize_search_patterns(self,
                                    task2keyword):
        """
        
        """
        ## Check Type
        if not isinstance(self._task2keyword, dict):
            raise TypeError("Expected task2keyword to be a dictionary.")
        ## Format and Compile
        patterns = {x:re.compile("|".join([rf"\b({re.escape(i)})\b" for i in y]), flags=re.IGNORECASE|re.UNICODE) for x, y in task2keyword.items()}
        ## Return
        return patterns

    def search(self,
               text):
        """
        
        """
        ## Initialize Cache
        matches = []
        ## Check Type
        if not isinstance(text, str):
            return matches
        ## Run Search
        for task, pattern in self._task2keyword_re.items():
            for match in pattern.finditer(text):
                ## Isolate Match and Normalize
                keyword = text[match.start():match.end()].lower()
                ## Cache
                matches.append({
                    "task":task,
                    "start":match.start(),
                    "end":match.end(), 
                    "keyword":keyword,
                })
        ## Return
        return matches

    def search_batch(self,
                     texts):
        """
        
        """
        ## Simple Map
        return list(map(self.search, texts))

class Tokenizer(object):

    """
    
    """

    def __init__(self,
                 replace_numeric=True,
                 filter_numeric=True,
                 replace_punc=True,
                 filter_punc=True,
                 negate_handling=True,
                 preserve_case=False):
        """
        
        """
        ## Class Tokenizer
        self._tokenizer = get_medical_tokenizer()
        ## Class Attributes
        self._replace_numeric = replace_numeric
        self._filter_numeric = filter_numeric
        self._replace_punc = replace_punc
        self._filter_punc = filter_punc
        self._negate_handling = negate_handling
        self._preserve_case = preserve_case

    def _add_negation(self,
                      tokens):
        """
        
        """
        ## Token Set
        negated_tokens = []
        ## Negation State
        negate_on = False
        ## Iterate Through Tokens
        for t in tokens:
            ## Check for Negation
            if t == "not" or t.endswith("n't"):
                negate_on = True
                continue
            ## Update Token Set
            if negate_on:
                negated_tokens.append(f"not_{t}")
                negate_on = False
            else:
                negated_tokens.append(t)
        return negated_tokens

    def _remove_from_text(self,
                          text,
                          rstring):
        """
        
        """
        ## Common Negation Patterns
        p = r"(cannot|not|\w*n't)?(\s)?\b{}\b".format(rstring)
        ## Replace
        text = re.sub(p, "", text)
        return text

    def tokenize(self,
                 text,
                 remove=set()):
        """
        
        """
        ## Cleaning
        for r in remove:
            text = self._remove_from_text(text, rstring=r)
        ## Split
        tokens = list(map(lambda i: i.text, self._tokenizer(text)))
        ## Casing
        if not self._preserve_case:
            tokens = list(map(lambda t: t.lower(), tokens))
        ## Format
        if self._filter_punc or self._replace_punc:
            is_punc = lambda x: (x in string.punctuation and x != "-") or all(char in string.punctuation for char in x)
            tokens = list(map(lambda i: "<PUNC>" if is_punc(i) else i, tokens))
        if self._filter_punc:
            tokens = list(filter(lambda i: i != "<PUNC>", tokens))
        if self._filter_numeric or self._replace_numeric:
            tokens = list(map(lambda i: "<NUM>" if any(c.isdigit() for c in i) else i, tokens))
        if self._filter_numeric:
            tokens = list(filter(lambda i: i != "<NUM>", tokens))
        ## Remove Empty
        tokens = list(filter(lambda i: len(i.strip()) > 0, tokens))
        ## Negate
        if self._negate_handling:
            tokens = self._add_negation(tokens)
        return tokens

class PhraseLearner(object):

    """
    
    """

    def __init__(self,
                 passes=0,
                 min_count=5,
                 threshold=1,
                 verbose=False):
        """
        
        """
        ## Working Space
        self._phrase_tuples = {i:{} for i in range(passes)}
        self._phrase_scores = {i:{} for i in range(passes)}
        ## Class Attributes
        self._passes = passes
        self._min_count = min_count
        self._threshold = threshold
        self._verbose = verbose

    def _dict2sparse(self,
                     dicts,
                     tardim):
        """
        
        """
        rows, cols, vals = [], [], []
        for row_ind, row_count in enumerate(dicts):
            for col_ind, value in row_count.items():
                rows.append(row_ind)
                cols.append(col_ind)
                vals.append(value)
        ## Ensure Completeness with Index
        for t in range(tardim):
            rows.append(row_ind + 1)
            cols.append(t)
            vals.append(1)
        X = sparse.csr_matrix((vals, (rows, cols)))
        X = X[:-1]
        return X

    def _count_cooccurences(self,
                            vocabulary,
                            vocabulary2ind,
                            tokens,
                            context_l,
                            context_r):
        """
        
        """
        ## Check Context Type
        if not isinstance(context_l, int) or not isinstance(context_r, int):
            raise TypeError("Context sizes (l,r) should be an integer.")
        ## Iterator
        if self._verbose:
            wrapper = tqdm(tokens, desc="[Counting Context Co-Occurrences]", total=len(tokens), file=sys.stdout)
        else:
            wrapper = tokens
        ## Counts/Co-occurence Matrix
        counts = np.zeros(len(vocabulary), dtype=int)
        coco_counts = [Counter() for _ in vocabulary]
        for toks in wrapper:
            if not toks:
                continue
            for t, tok in enumerate(toks):
                ## General Count
                if tok not in vocabulary2ind:
                    continue
                counts[vocabulary2ind[tok]] += 1
                ## Context Co-occurence
                context_toks = toks[max(0, t-context_l):t] + toks[t+1:t+1+context_r]
                for ct in context_toks:
                    if ct not in vocabulary2ind:
                        continue
                    coco_counts[vocabulary2ind[tok]][vocabulary2ind[ct]] += 1
        ## Convert to Sparse
        coco_counts = self._dict2sparse(coco_counts, tardim=len(vocabulary))
        return counts, coco_counts
    
    def _fit(self,
             tokens):
        """
        
        """
        ## Get Vocabulary
        vocabulary, vocabulary2ind = get_vocabulary(tokens,
                                                    rm_top=None,
                                                    min_freq=None,
                                                    max_freq=None,
                                                    stopwords=set())
        ## Counts
        counts, coco_counts = self._count_cooccurences(vocabulary,
                                                       vocabulary2ind,
                                                       tokens,
                                                       context_l=0,
                                                       context_r=1)
        ## Iterator
        if self._verbose:
            wrapper = tqdm(enumerate(coco_counts), total=coco_counts.shape[0], desc="[Identifying Phrases]", file=sys.stdout)
        else:
            wrapper = enumerate(coco_counts)
        ## Scoring and Phrase Extraction
        phrases = []
        for r, row in wrapper:
            ## Get Score
            numerator = (row.toarray() - self._min_count)[0] * len(vocabulary)
            denominator = counts[r] * counts
            rscore = np.divide(numerator, denominator)
            ## Phrase Isolation
            for c, s in enumerate(rscore):
                if s > self._threshold:
                    phrases.append([vocabulary[r], vocabulary[c], s])
        ## Sort Phrases
        phrases = sorted(phrases, key=lambda x: x[-1], reverse=True)
        ## Phrase Mapping and Scores
        phrase_tuples = {tuple(p[:2]):" ".join(p[:2]) for p in phrases}
        phrase_scores = {tuple(p[:2]):p[2] for p in phrases}
        return phrase_tuples, phrase_scores
    
    def fit(self,
            tokens):
        """
        
        """
        for epoch in range(self._passes):
            if self._verbose:
                print(f"[Beginning Phrase Learning Epoch {epoch+1}/{self._passes}")
            ## Get Phrase Tuples At Epoch
            self._phrase_tuples[epoch], self._phrase_scores[epoch] = self._fit(tokens)
            ## Transform Tokens (If More Epochs Remain)
            if epoch != self._passes - 1:
                tokens = self._transform(tokens, self._phrase_tuples[epoch])
        return self
    
    def fit_transform(self,
                      tokens):
        """
        
        """
        for epoch in range(self._passes):
            if self._verbose:
                print(f"[Beginning Phrase Learning Epoch {epoch+1}/{self._passes}")
            ## Get Phrase Tuples At Epoch
            self._phrase_tuples[epoch], self._phrase_scores[epoch] = self._fit(tokens)
            ## Transform Tokens (Always)
            tokens = self._transform(tokens, self._phrase_tuples[epoch])
        return tokens
        
    def _transform(self,
                   tokens,
                   phrase_tuples):
        """
        
        """
        ## Iterator
        if self._verbose:
            wrapper = tqdm(tokens, total=len(tokens), desc="[Rephrasing]", file=sys.stdout)
        else:
            wrapper = tokens
        ## Translate
        tokens_translated = []
        for toks in wrapper:
            i = 0
            toks_translated = []
            while i < len(toks):
                if i == len(toks) - 1:
                    toks_translated.append(toks[i])
                    break
                if (toks[i], toks[i+1]) in phrase_tuples:
                    toks_translated.append(phrase_tuples[(toks[i], toks[i+1])])
                    i += 2
                else:
                    toks_translated.append(toks[i])
                    i += 1
            tokens_translated.append(toks_translated)
        return tokens_translated
    
    def transform(self,
                  tokens):
        """
        
        """
        ## Apply Transformation Iteratively
        for epoch in range(self._passes):
            if self._verbose:
                print(f"[Applying Phrase Transformation {epoch+1}/{self._passes}")
            tokens = self._transform(tokens, self._phrase_tuples[epoch])
        return tokens
    
    def get_phrases(self):
        """
        
        """
        phrases = []
        for epoch, epoch_phrases in self._phrase_scores.items():
            for phrase, score in epoch_phrases.items():
                phrases.append([epoch, phrase[0], phrase[1], score])
        phrases = sorted(phrases, key=lambda x: x[-1], reverse=True)
        return phrases
