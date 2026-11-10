"""Compare le chargement d'un même checkpoint (ex. donut-base) entre deux versions de transformers.

1) Dans CHAQUE environnement :
     python compare_init_ckpt.py dump  /chemin/donut-base  dump_v4.pt     (env 6.3)
     python compare_init_ckpt.py dump  /chemin/donut-base  dump_v5.pt     (env 6.3.2)
2) Dans n'importe lequel des deux :
     python compare_init_ckpt.py compare dump_v4.pt dump_v5.pt
"""
import sys

import torch


def dump(ckpt, out):
    import transformers
    from transformers import VisionEncoderDecoderModel

    model, info = VisionEncoderDecoderModel.from_pretrained(ckpt, output_loading_info=True)
    model.eval()
    print("transformers", transformers.__version__)
    for k in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        print(f"{k:16s}: {len(info.get(k, []))}", list(info.get(k, []))[:10])

    # Entrée fixe, identique dans les deux environnements
    h, w = model.config.encoder.image_size if isinstance(model.config.encoder.image_size, (list, tuple)) \
        else (model.config.encoder.image_size,) * 2
    g = torch.Generator().manual_seed(0)
    pixel_values = torch.rand(1, 3, h, w, generator=g) * 2 - 1
    vocab = model.decoder.get_input_embeddings().weight.shape[0]
    decoder_input_ids = torch.randint(0, vocab, (1, 16), generator=g)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    with torch.no_grad():
        enc = model.encoder(pixel_values=pixel_values.to(dev)).last_hidden_state
        logits = model(pixel_values=pixel_values.to(dev), decoder_input_ids=decoder_input_ids.to(dev)).logits

    torch.save({
        "version": transformers.__version__,
        "state_dict": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
        "tied": model.decoder.get_output_embeddings().weight.data_ptr()
                == model.decoder.get_input_embeddings().weight.data_ptr(),
        "encoder_out": enc.float().cpu(),
        "logits": logits.float().cpu(),
    }, out)
    print("sauvé dans", out)


def compare(a_path, b_path, tol=1e-6):
    a, b = torch.load(a_path, weights_only=False), torch.load(b_path, weights_only=False)
    sa, sb = a["state_dict"], b["state_dict"]
    print(f"A = {a['version']}   B = {b['version']}")
    print("lm_head lié aux embeddings :", a["tied"], "/", b["tied"])
    only_a, only_b = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    print(f"clés seulement dans A : {len(only_a)}", only_a[:10])
    print(f"clés seulement dans B : {len(only_b)}", only_b[:10])

    diffs = []
    for k in sorted(set(sa) & set(sb)):
        if sa[k].shape != sb[k].shape:
            diffs.append((k, f"shape {tuple(sa[k].shape)} vs {tuple(sb[k].shape)}"))
        else:
            d = (sa[k] - sb[k]).abs().max().item()
            if d > tol:
                diffs.append((k, f"écart max {d:.2e}"))
    print(f"poids communs différents : {len(diffs)} / {len(set(sa) & set(sb))}")
    for k, msg in diffs[:20]:
        print("   ", k, "->", msg)

    for name in ("encoder_out", "logits"):
        d = (a[name] - b[name]).abs().max().item()
        print(f"écart max {name:11s}: {d:.2e}   (< 1e-3 : ok ; > 1e-2 : chargement ou calcul différent)")


if __name__ == "__main__":
    {"dump": dump, "compare": compare}[sys.argv[1]](*sys.argv[2:])
