import numpy as np
def update_roberta_keys(state, nlayer=24):
    for key in state.keys():
        if 'self_attn.q_proj' in key:
            return state
    new_dict = {}
    for key, val in state.items():
        if not 'self_attn.in_proj_' in key:
            new_dict[key] = val

    for i in range(nlayer):
        mhaw = 'decoder.sentence_encoder.layers.{}.self_attn.in_proj_weight'.format(i)
        mhab = 'decoder.sentence_encoder.layers.{}.self_attn.in_proj_bias'.format(i)
        weight = state[mhaw]
        bais = state[mhab]
        size = int(weight.size(0) / 3)
        # query, key, value
        qw = 'decoder.sentence_encoder.layers.{}.self_attn.q_proj.weight'.format(i)
        kw = 'decoder.sentence_encoder.layers.{}.self_attn.k_proj.weight'.format(i)
        vw = 'decoder.sentence_encoder.layers.{}.self_attn.v_proj.weight'.format(i)
        new_dict[qw] = weight[:size, : ]
        new_dict[kw] = weight[size:size * 2, : ]
        new_dict[vw] = weight[size * 2:, : ]

        # reconstruct weight
        rweight = np.concatenate((new_dict[qw].cpu().numpy(), new_dict[kw].cpu().numpy(), new_dict[vw].cpu().numpy()), axis=0)
        assert np.array_equal(rweight, weight.cpu().numpy())
        qb = 'decoder.sentence_encoder.layers.{}.self_attn.q_proj.bias'.format(i)
        kb = 'decoder.sentence_encoder.layers.{}.self_attn.k_proj.bias'.format(i)
        vb = 'decoder.sentence_encoder.layers.{}.self_attn.v_proj.bias'.format(i)
        new_dict[qb] = bais[:size]
        new_dict[kb] = bais[size:size * 2]
        new_dict[vb] = bais[size * 2:]
        rbais = np.concatenate((new_dict[qb].cpu().numpy(), new_dict[kb].cpu().numpy(), new_dict[vb].cpu().numpy()), axis=0)
        assert np.array_equal(rbais, bais.cpu().numpy())
    return new_dict

def patch_name_dict(state):
    new_state = {}
    for key, val in state.items():

        if key.startswith('decoder.sentence_encoder.emb'):
            key = key.replace("decoder.sentence_encoder", "bert.embeddings")
            key = key.replace("embed_tokens", "word_embeddings")
            key = key.replace("embed_positions", "position_embeddings")
            key = key.replace("emb_layer_norm", "LayerNorm")
        elif key.startswith('decoder.sentence_encoder.layers'):
            key = key.replace("decoder.sentence_encoder.layers", "bert.encoder.layer")
            key = key.replace("self_attn.k_proj", "attention.self.key")
            key = key.replace("self_attn.q_proj", "attention.self.query")
            key = key.replace("self_attn.v_proj", "attention.self.value")
            key = key.replace("self_attn.out_proj", "attention.output.dense")
            key = key.replace("self_attn_layer_norm", "attention.output.LayerNorm")
            key = key.replace("fc1", "intermediate.dense")
            key = key.replace("fc2", "output.dense")
            key = key.replace("final_layer_norm", "output.LayerNorm")
        new_state[key] = val

    return new_state
