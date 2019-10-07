# provides methods to process and display tokenized corpora
import urllib
from nltk.corpus import wordnet as wn

def val_if_gt(val):
    '''Returns the value, but only if it's a grammar subtoken'''
    return val if is_grammar(val) else None


def val_if_concept(val):
    '''Returns the value, but only if it's a concept subtoken'''
    return val if is_concept(val) else None


def is_grammar(val):
    '''Returns true if the string value is a grammar subtoken'''
    return val.startswith('GT_')


def is_lemma(val):
    '''Returns true if the string value is a lemma subtoken'''
    return val.startswith('lem_')


def is_concept(val):
    '''Returns true if the string value is a synset subtoken'''
    return val.startswith('wn31_')


def tlgs_subtok(sub_tokens, subtok_type):
    '''Returns the subtoken of the requested subtok_type or None
    Processes the list of sub_tokens and returns the subtoken of
    the requested subtok_type, if available. Otherwise, returns None'''
    if sub_tokens is None:
        raise ValueError("No sub_tokens passed", sub_tokens)
    if len(sub_tokens) < 2:
        raise ValueError("Expecting at least 2 sub token values, but found",
                         sub_tokens)
    if subtok_type == 't':
        return sub_tokens[0].replace('+', ' ')
    elif subtok_type == 'l':
        lem = sub_tokens[1] if is_lemma(sub_tokens[1]) else None
        return None if lem is None else lem.replace('lem_', '').replace('+', ' ')
    elif subtok_type == 'g':
        end = min(3, len(sub_tokens))
        vals = [val_if_gt(val) for val in sub_tokens[1:end]]
        gt_vals = [v.replace('GT_', '') for v in vals if v is not None]
        return gt_vals[0] if len(gt_vals) > 0 else None
    elif subtok_type == 's':
        end = min(4, len(sub_tokens))
        vals = [val_if_concept(val) for val in sub_tokens[1:end]]
        s_vals = [v for v in vals if v is not None]
        return s_vals[0] if len(s_vals) > 0 else None
    else:
        raise ValueError('Subtok type %s not supported for format tlgs' %
                         subtok_type)


def sub_tok(sub_tokens, subtok_type, expected_format='tlgs'):
    '''Returns the subtoken of the requested type, for the expected format'''
    if expected_format == 'tlgs':
        return tlgs_subtok(sub_tokens, subtok_type)
    else:
        raise ValueError('unsupported expected_format %s' % expected_format)


def subtok_dictionary(sub_tokens, line, include_glossa=True):
    '''Returns the list of sub_tokens as a python dictionary'''
    syn_tok = as_nltk_wnet_synset_id(sub_tok(sub_tokens, 's'))
    return {
        't': sub_tok(sub_tokens, 't'),
        'l': sub_tok(sub_tokens, 'l'),
        'g': sub_tok(sub_tokens, 'g'),
        's': syn_tok,
        'glossa': wnet_glossa(syn_tok),
        'line': line
    }


def as_nltk_wnet_synset_id(java_synset_id):
    '''Converts the tokenized synset id to the format expected by nltk'''
    if java_synset_id is None:
        return None
    else:
        return java_synset_id.replace('wn31_', '').replace('+', '_')


def wnet_glossa(wnet_synset_id):
    '''Retrieve the WordNet glossa for a given synset id'''
    if wnet_synset_id is None:
        return None
    try:
        synset = wn.synset(wnet_synset_id)
        return synset.definition()
    except Exception as e:
        print('Error retrieving synset for ', wnet_synset_id)
        return None


def open_as_token_dicts(file_name, token_format='tlgs',
                        max_lines=None, max_toks_per_line=None,
                        include_glossa=True):
    '''Returns a list of token dictionaries'''
    toked_lines = []
    with open(file_name, encoding='utf8') as f:
        lnum = 1
        for line in f.readlines():
            toks = [tok for tok in line.split()]
            end = len(toks)
            if max_toks_per_line is not None:
                end = min(max_toks_per_line, end)

            dec_toks = [urllib.parse.unquote(tok) for tok in toks]
            subdec_toks = [dec_tok.split(sep='|') for dec_tok in dec_toks]
            subtok_dics = [subtok_dictionary(subdec_tok,
                                             line=lnum,
                                             include_glossa=include_glossa)
                           for subdec_tok in subdec_toks]
            toked_lines = toked_lines + subtok_dics
            lnum += 1
            if (max_lines is not None) and lnum > max_lines:
                break
    return toked_lines
