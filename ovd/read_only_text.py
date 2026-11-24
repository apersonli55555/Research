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
        self.capture = False

    def _hook(self, mod, args, kwargs):
        if self.capture:
            self.saved.append({k: v for k, v in kwargs.items()})

    def fixed(self, img, F):
        g = self.g; self.saved = []; self.capture = True
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
