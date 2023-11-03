import joblib
import numpy as np
import torch

from pdf_difficulty.bootleg_gpt2 import GPTConfig, GPT
from pdf_difficulty.model import gpt2_classiffier_2_multi

def get_backbone(bscore):
    first_layer = "2fc"
    gptconf = GPTConfig(block_size=256, vocab_size=509, n_layer=12, n_head=12, n_embd=768, dropout=0.0,
                        first_layer=first_layer, bias=False, weight_loss=5)
    model = GPT(gptconf)

    checkpoint = torch.load(f"models/gpt2-base-fc-real/ckpt.pt",
                            map_location=torch.device('cpu'))
    # get the model
    model.load_state_dict(checkpoint['model'])


    get_embedding = model.get_embedding2
    bscore_matrix = bscore
    print(f"Input: {bscore_matrix.shape}")

    if bscore_matrix.shape[0] < 256:
        embed = model.get_embedding2(bscore_matrix.unsqueeze(0)).squeeze(0)
        print(f"Output: {embed.shape}")
    else:
        first_fragment = True
        embed = []
        for idx in range(0, bscore_matrix.shape[0] - 128, 128):
            jdx = idx + 256 if idx + 256 < bscore_matrix.shape[0] else bscore_matrix.shape[0]
            if first_fragment:
                embed.append(get_embedding(bscore_matrix[idx:jdx].unsqueeze(0)).squeeze(0).detach().cpu())
                first_fragment = False
            else:
                embed.append(get_embedding(bscore_matrix[idx + 128:jdx].unsqueeze(0)).squeeze(0).detach().cpu())
            # print([e.shape for e in embed])
        embed = torch.cat(embed)
        print(f"Output: {embed.shape}")
        assert embed.shape[0] == bscore_matrix.shape[0], f"{embed.shape[0]} != {bscore_matrix.shape[0]}"
    return embed




def get_latent_space(feat):
    feat = get_backbone(feat)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cpu'
    model = gpt2_classiffier_2_multi(
        checkpoint_dir=f"../../../bootlegGPT/models_trained/pt2-imslp-ft_fc5_real", device=device
    )
    checkpoint = torch.load(
        f"diff_model/models/multi_weighted_per_dataset/checkpoint_0.pth",
        map_location=torch.device('cpu')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    embedding = model.get_projection(feat.unsqueeze(0), torch.Tensor([feat.shape[0]]).int())
    ans = embedding.detach().numpy()
    # load numpy pca pca_two_dimensions_model.pkl and transform the data with joblib
    pca = joblib.load('diff_model/pca_two_dimensions_model.pkl')
    predicted_map = pca.transform(ans)[0]
    return predicted_map[0], predicted_map[1]

