
####################
### Imports
####################

## Standard Library
import re
from collections import Counter

## External
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

