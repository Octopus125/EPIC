import random
import numpy as np
import torch
import torch.nn.functional as F

# from utils.constant import aa_count_freq

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# def index_to_onehot(x, num_classes=20):
#     x = torch.tensor(x)

#     assert x.max().item() < num_classes, \
#         f'Error: {x.max().item()} >= {num_classes}'

#     x_onehot = F.one_hot(x, num_classes)
#     permute_order = (0, -1) + tuple(range(1, len(x.size())))
#     x_onehot = x_onehot.permute(permute_order)

#     return x_onehot.float()


# def logit_to_index(logit_p, random_state=False):
#     if random_state:
#         D = torch.distributions.Categorical(logit_p)
#         token_index = D.sample()
#     else:
#         token_index = logit_p.argmax(dim=-1)

#     return token_index

# def get_batch_info(data_list, device):
#     batch = []

#     for i in range(len(data_list)):
#         data_length = len(data_list[i])
#         data_value = torch.full([data_length], i, device=device)
#         batch.append(data_value)

#     batch = torch.concat(batch)

#     return batch

# def get_seq_noise(seq_len=1, device=None, noise_state="dmd", n_class=20):
#     # dud: discrete uniform distribution
#     # dmd: discrete marginal distribution

#     if noise_state == "dud":
#         noise = torch.ones([seq_len, n_class], device=device) / n_class
#     else:
#         noise = torch.tensor(aa_count_freq, device=device, dtype=torch.float32).unsqueeze(dim=0).repeat(seq_len, 1)

#     return noise


# # Qt = alphas_bar * I + (1 - alphas_bar) * K
# def get_Qt_weight(alphas_bar, noise, batch, device, n_class=20):
#     # Q_weight = [bar_t * torch.eye(self.n_class, device=self.device) + (1 - bar_t) * noise for bar_t in
#     #            token_alphas_bar]
#     # Q_weight = torch.stack(Q_weight).float()
#     Qt_weight = [bar_t * torch.eye(n_class, device=device) + (1 - bar_t) * noise for bar_t in
#                  alphas_bar]
#     Qt_weight = torch.stack(Qt_weight).float() # [batch size,20,20]
#     Qt_weight = Qt_weight.index_select(0, batch)
#     # [N,20,20]
#     return Qt_weight

# def get_para_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timestep, device):
#     def sigmoid(x):
#         return 1 / (np.exp(-x) + 1)

#     if beta_schedule == "quad":
#         betas = (
#                 np.linspace(
#                     beta_start ** 0.5,
#                     beta_end ** 0.5,
#                     num_diffusion_timestep,
#                     dtype=np.float64,
#                 )
#                 ** 2
#         )
#     elif beta_schedule == "linear":
#         betas = np.linspace(
#             beta_start, beta_end, num_diffusion_timestep, dtype=np.float64
#         )
#     elif beta_schedule == "const":
#         betas = beta_end * np.ones(num_diffusion_timestep, dtype=np.float64)
#     elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
#         betas = 1.0 / np.linspace(
#             num_diffusion_timestep, 1, num_diffusion_timestep, dtype=np.float64
#         )
#     elif beta_schedule == "sigmoid":
#         betas = np.linspace(-6, 6, num_diffusion_timestep)
#         betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
#     else:
#         raise NotImplementedError(beta_schedule)

#     assert betas.shape == (num_diffusion_timestep,)

#     betas = torch.tensor(betas, device=device).float()
#     alphas = (1. - betas)
#     alphas_bar = (1. - betas).cumprod(dim=0)

#     return betas, alphas, alphas_bar