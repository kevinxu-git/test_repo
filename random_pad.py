"""Remplace l'option random_padding=True de DonutProcessor (supprimée en transformers 5.x).

Usage dans le __getitem__ du Dataset :
    pixel_values = donut_pixel_values(processor, image, train=(self.split == "train"))
"""
import random

import torch
import torch.nn.functional as F


def random_pad(pixel_values, height, width, train=True, pad_value=-1.0):
    """Place l'image (3, h, w) déjà normalisée dans un canevas (3, height, width).

    train=True  : position aléatoire (équivalent de random_padding=True en 4.37)
    train=False : image centrée (comportement par défaut, à garder pour la validation)
    pad_value=-1 : pixel noir après normalisation Donut (mean = std = 0.5)
    """
    dh, dw = height - pixel_values.shape[-2], width - pixel_values.shape[-1]
    assert dh >= 0 and dw >= 0, "l'image doit être redimensionnée avant le padding (do_pad=False)"
    top = random.randint(0, dh) if train else dh // 2
    left = random.randint(0, dw) if train else dw // 2
    return F.pad(pixel_values, (left, dw - left, top, dh - top), value=pad_value)


def donut_pixel_values(processor, image, train=True):
    ip = processor.image_processor
    pv = ip(image, do_pad=False, return_tensors="pt").pixel_values[0]  # redimensionnée, normalisée, sans padding
    return random_pad(pv, ip.size["height"], ip.size["width"], train=train)
