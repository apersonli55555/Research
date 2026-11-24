"""Expert idea 2, variant B: class text tokens READ the fixed pass's per-layer visual states, never write to them.

ReadOnlyText(g).fixed(img, F) runs the fixed prompt once, saving every encoder layer's inputs; .class_tokens(c) then
replays only the text side of each encoder layer (fusion text<-image update + per-phrase text self-attention) for "c ."
against the saved visual states. The decoder queries/boxes come from the fixed pass. Invariant by construction.
"""
import torch
from transformers.models.grounding_dino.modeling_grounding_dino import generate_masks_with_special_tokens_and_transfer_map as gm


class ReadOnlyText:
    def __init__(self, g):
        self.g, self.m = g, g.model
        self.saved = []
        for layer in self.m.model.encoder.layers:
            layer.register_forward_pre_hook(self._hook, with_kwargs=True)
        self.m.model.decoder.register_forward_pre_hook(self._dhook, with_kwargs=True)
        self.capture = False; self.dec = None

    def _dhook(self, mod, args, kwargs):
        if self.capture:
            self.dec = dict(kwargs); self.dec_args = args

    def _hook(self, mod, args, kwargs):
        if self.capture:
            self.saved.append({k: v for k, v in kwargs.items()})

    def fixed(self, img, F):
        g = self.g; self.saved = []; self.capture = True; self.dec = None
        x = g.proc(images=img, text=" . ".join(F) + " .", return_tensors="pt").to(g.dev)
        try:
            with torch.no_grad():
                o = self.m(**x, output_hidden_states=False)
        finally:
            self.capture = False
        self.out = o
        return o

    def text_inputs(self, phrase):
        g = self.g
        x = g.proc.tokenizer(phrase, return_tensors="pt").to(g.dev)
        mask3, pos = gm(x["input_ids"])
        m4 = mask3[:, None] if mask3.ndim == 3 else mask3
        with torch.no_grad():
            h = self.m.model.text_backbone(x["input_ids"], m4, x.get("token_type_ids"), pos, return_dict=True).last_hidden_state
            t = self.m.model.text_projection(h)
        tam = ~x["attention_mask"].bool()
        ref = self.saved[0]["text_self_attention_masks"]
        inv = ~mask3                                   # the model passes ~mask to the encoder (GroundingDinoModel.forward)
        sm = inv if ref.ndim == inv.ndim else inv[:, None]
        return t, tam, sm, pos

    def replay(self, t, tam, sm, pos):
        """Run only the text side of every encoder layer against the saved visual states."""
        with torch.no_grad():
            for layer, kw in zip(self.m.model.encoder.layers, self.saved):
                tpe = layer.get_text_position_embeddings(t, kw.get("text_position_embedding"), pos)
                _, (t, _) = layer.fusion_layer(vision_features=kw["vision_features"], text_features=t,
                                               attention_mask_vision=kw["key_padding_mask"], attention_mask_text=tam)
                t, _ = layer.text_enhancer_layer(hidden_states=t, attention_masks=~sm, position_embeddings=tpe)
        return t

    def class_tokens(self, c):
        t, tam, sm, pos = self.text_inputs(c + " .")
        t = self.replay(t, tam, sm, pos)[0]
        return t[1:1 + len(self.g.tokens(c))]

    def decode(self, text_states, text_mask_real):
        """Re-run ONLY the decoder from the fixed pass's inputs, with `text_states` as the text it cross-attends to.
        Returns (normalised cxcywh boxes, logits over text tokens) of the last decoder layer."""
        kw = dict(self.dec)
        for k in list(kw):
            if k.startswith("text_encoder_hidden_states") or k == "text_encoder_hidden_states":
                kw[k] = text_states
            if k == "text_encoder_attention_mask":
                kw[k] = ~text_mask_real if kw[k].dtype == torch.bool else kw[k]
        with torch.no_grad():
            d = self.m.model.decoder(*self.dec_args, **kw)
            hs, refs = d.intermediate_hidden_states, d.intermediate_reference_points
            L = hs.shape[1] - 1
            ref = torch.special.logit(refs[:, L - 1], eps=1e-5)
            box = (self.m.bbox_embed[L](hs[:, L]) + ref).sigmoid()[0]
            logit = (hs[:, L] @ text_states.transpose(-1, -2))[0]
        return box, logit

    def class_decode(self, c):
        """Variant C: class tokens replayed read-only through the encoder, then a decoder pass reading only them."""
        t, tam, sm, pos = self.text_inputs(c + " .")
        T = self.replay(t, tam, sm, pos)
        box, logit = self.decode(T, ~tam)
        n = len(self.g.tokens(c))
        return box, logit[:, 1:1 + n]
