from __future__ import absolute_import, division, print_function

import collections
import unicodedata

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    idx = 0
    with open(vocab_file, "r") as f:
        while True:
            token = f.readline().strip()
            if token == "":
                break
            vocab[token] = idx
            idx += 1
    return vocab

def convert_tokens_to_ids(vocab, tokens):
    # Generate ids from tokens
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids

def whitespace_tokenize(text):
    # Simple tokenizer using space
    tokens = text.strip().split(' ')
    return tokens

def _is_whitespace(char):
    if char in [" ", "\t", "\n", "\r"]:
        return True
    return False

def _is_control(char):
    # From original authors: These are technically control characters but we count them as whitespace
    if char in ["\t", "\n", "\r"]:
        return False

    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    cp = ord(char)
    # From original authors: treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    else:
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
    return False

class FullTokenizer():
    # This is the class for end-to-end Tokenization
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)


class BasicTokenizer():
    # Class for basic tokenization, spliting and lowercasing
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    # first split text on space, and then split tokens on punctuations
    def tokenize(self, text):
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        return split_tokens

    def _run_strip_accents(self, text):
        # strip accents in text
        text = unicodedata.normalize("NFD", text)
        output = ""
        for char in text:
            cat = unicodedata.category(char)
            if cat != "Mn":
                output += char
        return output

    def _run_split_on_punc(self, text):
        # splits on punctuation in text, output: [text, punc, text ...]
        chars = list(text)
        start_new_word = True
        output = []
        for char in chars:
            if _is_punctuation(char):
                output.append(char)
                start_new_word = True
            else:
                if start_new_word:
                    output.append('')
                    start_new_word = False
                output[-1] += char

        return output

    def _clean_text(self, text):
        # Remove invalid character and whitespace
        output = ""
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                pass
            elif _is_whitespace(char):
                output += " "
            else:
                output += char
        return output


# From original authors
class WordpieceTokenizer():

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        # text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
