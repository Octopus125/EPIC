def esm_decode_batch(alphabet, batch_tokens):
    cls_idx = alphabet.cls_idx
    pad_idx = alphabet.padding_idx
    eos_idx = alphabet.eos_idx
    
    seq_list = []
    decode_mask = batch_tokens.ne(pad_idx) & batch_tokens.ne(cls_idx) & batch_tokens.ne(eos_idx)
    for i in range(batch_tokens.shape[0]):
        decode_tokens = batch_tokens[i, decode_mask[i]]
        decode_seq = [alphabet.get_tok(decode_tokens[j]) for j in range(len(decode_tokens))]
        decode_seq = ''.join(decode_seq)
        seq_list.append(decode_seq)
    return seq_list
    
def esm_forward2(self, padding_mask, embeded_tokens, repr_layers=[], need_head_weights=False):
    x = self.embed_scale * embeded_tokens

    if self.token_dropout:
        mask_ratio_train = 0.15 * 0.8
        x = x * (1 - mask_ratio_train)

    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    # (B, T, E) => (T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=need_head_weights,
        )
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

    # last hidden representation should have layer norm applied
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    return result