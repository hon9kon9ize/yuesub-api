import os
import glob
import json
import re
import unicodedata
import argparse
from collections import Counter, defaultdict
from opencc import OpenCC
import difflib

# Initialize OpenCC converter (HK traditional --> simplified)
opencc = OpenCC("hk2s")

# Mixed-token regex: ASCII words or single CJK
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4E00-\u9FFF]")

def post_and_tokenize(text: str) -> list:
    """
    1) Convert HK traditional → simplified.
    2) Strip all Unicode punctuation.
    3) Find all ASCII‑word or single CJK tokens.
    4) Lower‑case any ASCII tokens.
    Returns list of tokens.
    """
    s = opencc.convert(text)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "P")
    raw_toks = TOKEN_PATTERN.findall(s)
    toks = [tok.lower() if tok.isascii() else tok for tok in raw_toks]
    return toks


def compute_mer_and_edits(ref_tokens: list, hyp_tokens: list):
    """
    Align ref_tokens vs hyp_tokens, returning substitution, deletion,
    insertion counts and list of substitution pairs.
    """
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
    substitutions = deletions = insertions = 0
    subs_pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            ref_span = ref_tokens[i1:i2]
            hyp_span = hyp_tokens[j1:j2]
            m = min(len(ref_span), len(hyp_span))
            substitutions += m
            deletions += len(ref_span) - m
            insertions += len(hyp_span) - m
            for k in range(m):
                subs_pairs.append((ref_span[k], hyp_span[k]))
        elif tag == 'delete':
            deletions += (i2 - i1)
        elif tag == 'insert':
            insertions += (j2 - j1)
    total_tokens = len(ref_tokens)
    return substitutions, deletions, insertions, total_tokens, subs_pairs


def process_file(path: str, top_n: int):
    """
    Compute and print MER metrics for a single file.
    """
    dataset_metrics = defaultdict(lambda: {'sub': 0, 'del': 0, 'ins': 0, 'tok': 0})
    subs_counter = defaultdict(Counter)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            ds = rec.get('dataset', 'unknown')
            r_tokens = post_and_tokenize(rec.get('reference', ''))
            h_tokens = post_and_tokenize(rec.get('prediction', ''))
            sub, dele, ins, tok, subs = compute_mer_and_edits(r_tokens, h_tokens)
            dataset_metrics[ds]['sub'] += sub
            dataset_metrics[ds]['del'] += dele
            dataset_metrics[ds]['ins'] += ins
            dataset_metrics[ds]['tok'] += tok
            subs_counter[ds].update(subs)

    print(f"File: {os.path.basename(path)}")
    print("==============================")

    # Per-dataset breakdown
    overall = {'sub': 0, 'del': 0, 'ins': 0, 'tok': 0}
    for ds, m in dataset_metrics.items():
        sub, dele, ins, tok = m['sub'], m['del'], m['ins'], m['tok']
        mer = (sub + dele + ins) / tok if tok > 0 else 0.0
        print(f"Dataset: {ds}")
        print(f"  MER:           {mer:.2%}")
        print(f"  Substitutions: {sub}")
        print(f"  Deletions:     {dele}")
        print(f"  Insertions:    {ins}")
        print(f"  Tokens:        {tok}")
        print("  --------------------------")
        overall['sub'] += sub
        overall['del'] += dele
        overall['ins'] += ins
        overall['tok'] += tok

    # Overall for file
    o_sub, o_del, o_ins, o_tok = (
        overall['sub'], overall['del'], overall['ins'], overall['tok']
    )
    overall_mer = (o_sub + o_del + o_ins) / o_tok if o_tok > 0 else 0.0
    print("Overall:")
    print(f"  MER:           {overall_mer:.2%}")
    print(f"  Substitutions: {o_sub}")
    print(f"  Deletions:     {o_del}")
    print(f"  Insertions:    {o_ins}")
    print(f"  Tokens:        {o_tok}")

    # Top substitutions
    overall_subs = Counter()
    for ctr in subs_counter.values():
        overall_subs.update(ctr)
    print("\nTop replacements:")
    for (r, h), cnt in overall_subs.most_common(top_n):
        print(f"  '{r}' → '{h}': {cnt}")


def main(top_n: int):
    # Hardcoded directory
    dir_path = 'output'
    jsonl_files = sorted(glob.glob(os.path.join(dir_path, '*.jsonl')))
    if not jsonl_files:
        print(f"No .jsonl files found in '{dir_path}'")
        return

    for idx, path in enumerate(jsonl_files):
        process_file(path, top_n)
        if idx < len(jsonl_files) - 1:
            print('\n' + '='*40 + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute MER for each JSONL file under the output/ directory.')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top replacements to show')
    args = parser.parse_args()
    main(args.top_n)
