#!/usr/bin/env python3
"""
Test du tokenizer Donut sur du chinois traditionnel.

Usage :
    python test_tokenizer_donut.py corpus.txt [--checkpoint naver-clova-ix/donut-base] [--max-len 768]

corpus.txt : une page (ou un JSON cible serialise) par ligne, en chinois traditionnel.
Idealement la verite terrain annotee ; a defaut, la sortie PP-OCR sur ~50 pages.

Dependances : transformers, sentencepiece, opencc-python-reimplemented (optionnel)
"""
import argparse
from collections import Counter

from transformers import AutoConfig, AutoTokenizer

EXEMPLES_TW = [
    "臺北市信義區市府路一段",
    "台北市中正區重慶南路三段12號之3",
    "身分證字號：A123456789",
    "統一編號：04595257",
    "中華民國一一三年五月二十日",
    "新臺幣壹萬貳仟參佰元整",
    "負責人（大小章）「用印」",
    "１２３號５樓",
]


def charger_opencc():
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    for conf in ("t2s", "t2s.json"):
        try:
            return OpenCC(conf)
        except Exception:
            continue
    return None


def analyser(pages, tok, max_len):
    freq = Counter(c for p in pages for c in p if not c.isspace())
    total = sum(freq.values())
    unk_id = tok.unk_token_id

    # Test caractere par caractere : inconnus vs modifies au decodage
    inconnus, modifies = Counter(), {}
    for c, n in freq.items():
        ids = tok(c, add_special_tokens=False).input_ids
        if unk_id in ids:
            inconnus[c] = n
            continue
        rendu = tok.decode(ids, skip_special_tokens=True).strip()
        if rendu != c:
            modifies[c] = (rendu, n)

    longueurs = sorted(len(tok(p, add_special_tokens=False).input_ids) for p in pages)

    def pct(q):
        return longueurs[min(len(longueurs) - 1, int(q * len(longueurs)))]

    return {
        "chars_uniques": len(freq),
        "occurrences": total,
        "taux_unk": sum(inconnus.values()) / total,
        "inconnus": inconnus,
        "modifies": modifies,
        "fertilite": sum(longueurs) / total,
        "p50": pct(0.5),
        "p95": pct(0.95),
        "max": longueurs[-1],
        "depassements": (sum(l > max_len for l in longueurs) / len(longueurs)) if max_len else None,
    }


def rapport(titre, r, max_len, top=30):
    print(f"\n=== {titre} ===")
    print(f"Caracteres uniques : {r['chars_uniques']} | occurrences : {r['occurrences']}")
    print(f"Occurrences en <unk> : {r['taux_unk']:.3%} ({len(r['inconnus'])} caracteres distincts)")
    if r["inconnus"]:
        print("  Plus frequents :", " ".join(f"{c}x{n}" for c, n in r["inconnus"].most_common(top)))
    print(f"Caracteres modifies au decodage : {len(r['modifies'])}")
    for c, (rendu, n) in sorted(r["modifies"].items(), key=lambda x: -x[1][1])[:top]:
        print(f"  {c!r} (U+{ord(c):04X}) -> {rendu!r}  x{n}")
    print(f"Fertilite : {r['fertilite']:.2f} token/caractere")
    print(f"Longueur par ligne (tokens) : p50={r['p50']} p95={r['p95']} max={r['max']}")
    if max_len:
        print(f"Lignes depassant {max_len} tokens : {r['depassements']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--checkpoint", default="naver-clova-ix/donut-base")
    ap.add_argument("--max-len", type=int, default=None,
                    help="defaut : max_position_embeddings du decodeur")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    max_len = args.max_len
    if max_len is None:
        try:
            max_len = AutoConfig.from_pretrained(args.checkpoint).decoder.max_position_embeddings
        except Exception:
            max_len = None

    with open(args.corpus, encoding="utf-8") as f:
        pages = [l.strip() for l in f if l.strip()]

    print(f"Tokenizer : {type(tok).__name__} | vocab : {len(tok)} | max_len : {max_len}")
    rapport("Corpus traditionnel", analyser(pages, tok, max_len), max_len)

    cc = charger_opencc()
    if cc:
        r_s = analyser([cc.convert(p) for p in pages], tok, max_len)
        print("\n=== Meme corpus converti en simplifie (mesure du biais) ===")
        print(f"Occurrences en <unk> : {r_s['taux_unk']:.3%} | fertilite : {r_s['fertilite']:.2f}")
    else:
        print("\n(opencc absent : comparaison avec le simplifie ignoree)")

    print("\n=== Exemples taiwanais ===")
    for ex in EXEMPLES_TW:
        ids = tok(ex, add_special_tokens=False).input_ids
        rendu = tok.decode(ids, skip_special_tokens=True)
        flag = "OK  " if rendu.replace(" ", "") == ex.replace(" ", "") else "DIFF"
        print(f"[{flag}] {ex}")
        print(f"       {tok.convert_ids_to_tokens(ids)}")
        print(f"       -> {rendu}")


if __name__ == "__main__":
    main()
