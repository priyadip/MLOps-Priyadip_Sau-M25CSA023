"""
Q1  BLEU evaluation using sacrebleu.

Expects:
  output.txt     (generated hypotheses, one per line)
  reference.txt  (human references, one per line, SAME ORDER)
"""

from pathlib import Path
import sacrebleu


def load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    hyps = load_lines(Path("output.txt"))
    refs = load_lines(Path("reference.txt"))

    if len(hyps) != len(refs):
        print(f"[warn] hyps={len(hyps)}  refs={len(refs)} — truncating to min")
        n = min(len(hyps), len(refs))
        hyps, refs = hyps[:n], refs[:n]

    # sacrebleu corpus_bleu takes list of hypotheses and list of reference-lists
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(bleu)
    print(f"\nBLEU score: {bleu.score:.4f}")


if __name__ == "__main__":
    main()
