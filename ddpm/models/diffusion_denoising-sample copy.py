from typing import Optional, Tuple, cast, Union
import logging

import torch
import torch as th
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import math

from torch.distributions import Categorical

from .one_hot_categorical import OneHotCategoricalBCHW

LOGGER = logging.getLogger(__name__)

__all__ = ["DiffusionModel", "DenoisingModel"]


def linear_schedule(time_steps: int, start=1e-2, end=0.2) -> Tuple[Tensor, Tensor, Tensor]:
    betas = torch.linspace(start, end, time_steps)
    alphas = 1 - betas
    cumalphas = torch.cumprod(alphas, dim=0)
    return betas, alphas, cumalphas


def cosine_schedule(time_steps: int, s: float = 8e-3) -> Tuple[Tensor, Tensor, Tensor]:
    t = torch.arange(0, time_steps)
    s = 0.008
    cumalphas = torch.cos(((t / time_steps + s) / (1 + s)) * (math.pi / 2)) ** 2

    def func(t): return math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2

    betas_ = []
    for i in range(time_steps):
        t1 = i / time_steps
        t2 = (i + 1) / time_steps
        betas_.append(min(1 - func(t2) / func(t1), 0.999))
    betas = torch.tensor(betas_)
    alphas = 1 - betas
    return betas, alphas, cumalphas

def build_piecewise_Qs(time_steps: int) -> torch.Tensor:
    r"""
    使用“权重插值”的方式，在整个 time_steps 范围内平滑地混合：
      1) I (单位矩阵，保持类别)
      2) T_rel (你给定的关系矩阵)
      3) U (均匀矩阵)

    示例策略（可按需修改）：
      - 在前半段，让 Q^(t) 从 I 逐渐过渡到 T_rel
      - 在后半段，让 Q^(t) 从 T_rel 逐渐过渡到 U

    数学写法（举例）：
      令 ratio = t / (time_steps - 1)，遍历 t=0..time_steps-1:
      若 ratio < 0.5:
         ratio2 = 2 * ratio
         a(t) = 1 - ratio2    # 对应 I 的权重
         b(t) = ratio2        # 对应 T_rel 的权重
         c(t) = 0
      否则:
         ratio2 = (ratio - 0.5)/0.5
         a(t) = 0
         b(t) = 1 - ratio2    # 对应 T_rel
         c(t) = ratio2        # 对应 U

      然后:
         Q^(t) = a(t)*I + b(t)*T_rel + c(t)*U
      并对行做归一化，以确保它是一个转移矩阵 (马尔可夫)

    :param time_steps: 扩散总步数 (如 100, 200 等)
    :return: shape=[time_steps, C, C], 对应每个时间步的离散转移矩阵
    """
    betas, _, _ = cosine_schedule(time_steps)
    # 你给定的关系矩阵(13x13示例)；若要改大小，请自行替换
    # T_rel = torch.tensor([
    #     [1.0, 0.9103944, 0.9360034, 0.7782025, 0.8140766, 0.9339747, 0.8226238, 0.72960496, 0.50996536, 0.8906735, 0.93585116, 0.64702666, 0.60964507],
    #     [0.9103944, 1.0, 0.8096776, 0.4478394, 0.4882328, 0.95553786, 0.65318304, 0.52245796, 0.23281312, 0.84671295, 0.80021423, 0.37156302, 0.71240836],
    #     [0.9360034, 0.8096776, 0.99999976, 0.7853691, 0.8048417, 0.88164884, 0.8498865, 0.7543824, 0.60554063, 0.86240876, 0.9897057, 0.7070847, 0.5643366],
    #     [0.7782025, 0.4478394, 0.7853691, 1.0, 0.9620664, 0.571296, 0.69611734, 0.649873, 0.6131677, 0.59531164, 0.7999625, 0.76997674, 0.19369236],
    #     [0.8140766, 0.4882328, 0.8048417, 0.9620664, 0.99999976, 0.5951507, 0.7166575, 0.68476063, 0.62400615, 0.61899745, 0.8191158, 0.7602266, 0.20559905],
    #     [0.9339747, 0.95553786, 0.88164884, 0.571296, 0.5951507, 0.99999976, 0.755376, 0.61093974, 0.38700944, 0.9048203, 0.8836339, 0.5147647, 0.6357617],
    #     [0.8226238, 0.65318304, 0.8498865, 0.69611734, 0.7166575, 0.755376, 0.9999996, 0.8964518, 0.6087455, 0.8672453, 0.85789853, 0.71752006, 0.3600229],
    #     [0.72960496, 0.52245796, 0.7543824, 0.649873, 0.68476063, 0.61093974, 0.8964518, 0.99999976, 0.6996984, 0.75485516, 0.74201214, 0.6122701, 0.27163282],
    #     [0.50996536, 0.23281312, 0.60554063, 0.6131677, 0.62400615, 0.38700944, 0.6087455, 0.6996984, 1.0, 0.510165, 0.5903994, 0.61400384, 0.0],
    #     [0.8906735, 0.84671295, 0.86240876, 0.59531164, 0.61899745, 0.9048203, 0.8672453, 0.75485516, 0.510165, 0.99999976, 0.86971223, 0.5595701, 0.5538537],
    #     [0.93585116, 0.80021423, 0.9897057, 0.7999625, 0.8191158, 0.8836339, 0.85789853, 0.74201214, 0.5903994, 0.86971223, 1.0, 0.7183207, 0.52154696],
    #     [0.64702666, 0.37156302, 0.7070847, 0.76997674, 0.7602266, 0.5147647, 0.71752006, 0.6122701, 0.61400384, 0.5595701, 0.7183207, 1.0, 0.04032332],
    #     [0.60964507, 0.71240836, 0.5643366, 0.19369236, 0.20559905, 0.6357617, 0.3600229, 0.27163282, 0.0, 0.5538537, 0.52154696, 0.04032332, 1.0],
    # ], dtype=torch.float32)

    T_rel = torch.tensor([[1.0000001,  0.79102033, 0.8146844,  0.7428482,  0.74617374, 0.7831711,
        0.7625799,  0.7265548,  0.62626827, 0.76917535, 0.78334457, 0.6551933,
        0.65992606],
        [0.79102033, 1.0,         0.9186988,  0.8209396,  0.84689295, 0.9077158,
        0.8838165,  0.83717036, 0.70907855, 0.8407782,  0.90452224, 0.76641697,
        0.79645807],
        [0.8146844,  0.9186988,  1.0,         0.886816,   0.8805472,  0.9359363,
        0.9126045,  0.88855517, 0.7851754,  0.88552105, 0.9398908,  0.83387434,
        0.7673819],
        [0.7428482,  0.8209396,  0.886816,   0.9999999,  0.8904724,  0.83010733,
        0.84834063, 0.82388854, 0.7273655,  0.77442014, 0.85312283, 0.7631718,
        0.7011433],
        [0.74617374, 0.84689295, 0.8805472,  0.8904724,  0.9999999,  0.8444572,
        0.8839303,  0.85328865, 0.72266877, 0.7781976,  0.8888509,  0.7625368,
        0.7319992],
        [0.7831711,  0.9077158,  0.9359363,  0.83010733, 0.8444572,  1.0,
        0.8825142,  0.84960115, 0.7111397,  0.9073856,  0.91913617, 0.795673,
        0.7585107],
        [0.7625799,  0.8838165,  0.9126045,  0.84834063, 0.8839303,  0.8825142,
        1.0,        0.90161,    0.7546347,  0.81079304, 0.9389398,  0.831022,
        0.76170397],
        [0.7265548,  0.83717036, 0.88855517, 0.82388854, 0.85328865, 0.84960115,
        0.90161,    1.0,        0.8524681,  0.7626467,  0.857942,   0.80697465,
        0.7069379],
        [0.62626827, 0.70907855, 0.7851754,  0.7273655,  0.72266877, 0.7111397,
        0.7546347,  0.8524681,  1.0,        0.6582587,  0.74424,    0.7320045,
        0.5923711],
        [0.76917535, 0.8407782,  0.88552105, 0.77442014, 0.7781976,  0.9073856,
        0.81079304, 0.7626467,  0.6582587,  0.9999999,  0.8468666,  0.7283509,
        0.7158224],
        [0.78334457, 0.90452224, 0.9398908,  0.85312283, 0.8888509,  0.91913617,
        0.9389398,  0.857942,   0.74424,    0.8468666,  1.0,        0.82826066,
        0.76576924],
        [0.6551933,  0.76641697, 0.83387434, 0.7631718,  0.7625368,  0.795673,
        0.831022,   0.80697465, 0.7320045,  0.7283509,  0.82826066, 0.99999994,
        0.6370567],
        [0.65992606, 0.79645807, 0.7673819,  0.7011433,  0.7319992,  0.7585107,
        0.76170397, 0.7069379,  0.5923711,  0.7158224,  0.76576924, 0.6370567,
        0.9999999]], dtype=torch.float32)
   
    # 1) 读取类别数 C
    C = T_rel.shape[0]

    # 2) 行归一化 (若你确信 T_rel 已归一化可省略)
    # T_rel = T_rel - torch.diag_embed(T_rel.diagonal())
    T_rel = T_rel / (T_rel.sum(dim=1, keepdim=True) + 1e-12)

    # 3) 构造基础矩阵 I, U
    I = torch.eye(C, dtype=torch.float32)
    U = torch.ones(C, C, dtype=torch.float32)/C
    

    Qs = []
    for t in range(time_steps):
        beta_t = betas[t].item()
        wI = 1.0 - beta_t
        wR = beta_t * (1.0 - beta_t)
        wU = beta_t**2

        # 三者混合
        Q_t = wI * I + wR * T_rel + wU * U
        # Q_t = (1.0 - beta_t) * I + beta_t * T_rel

        # 行归一化
        Q_t = Q_t / (Q_t.sum(dim=1, keepdim=True) + 1e-12)

        Qs.append(Q_t)


    # eps = 0.01
    # Qs = []
    # for t in range(time_steps):
    #     ratio = t / max(time_steps-1, 1e-9)  # in [0,1]
    #     if ratio < 0.5:
    #         ratio2 = ratio*2.0
    #         # 原先 wI=1-ratio2, wT=ratio2, wU=0
    #         # 改成:
    #         wU = eps                           # 给 U 一点噪声
    #         wI = (1.0 - ratio2) * (1.0 - eps)  # 主要份额
    #         wT = ratio2 * (1.0 - eps)          # 次要份额
    #     else:
    #         ratio2 = (ratio - 0.5)/0.5
    #         # 原先 wI=0, wT=1-ratio2, wU=ratio2
    #         # 改成:
    #         wU = ratio2*(1-eps)                # 或保留一点eps给 I
    #         wT = (1.0 - ratio2)*(1-eps)        
    #         wI = eps * 0.0  # 或者给 wI= eps (让I保留极微量)
    #         # 视你需求, 也可 wI= eps, wT= (1-ratio2)*(1-2*eps), wU= ...
        
    #     # 再保证它们之和=1? => wSum= wI+wT+wU;  (如你想精确)
    #     wSum = wI + wT + wU
    #     wI /= wSum
    #     wT /= wSum
    #     wU /= wSum

    #     Q_t = wI * I + wT * T_rel + wU * U
    #     Q_t = Q_t / (Q_t.sum(dim=1, keepdim=True)+1e-12)  # 行归一化

    #     Qs.append(Q_t)

    return torch.stack(Qs, dim=0)  # shape=[time_steps, C, C]


'''
self, schedule: str, time_steps: int, num_classes: int, schedule_params=None'''


class DiffusionModel(nn.Module):
    def __init__(self, schedule: str, time_steps: int, num_classes: int, schedule_params=None):
        super().__init__()
        self.time_steps = time_steps

        Qs = build_piecewise_Qs(time_steps)   # => [T, C, C]
        # Qs = Qs.to(torch.float32)
        self.register_buffer("Qs", Qs)        # => shape=[T,C,C]
        self.num_classes = Qs.shape[1]

        # 前缀乘积 Qs_cumprod[t] = Q^(1)*...*Q^(t+1)
        Qs_cum = torch.empty_like(Qs)
        Qs_cum[0] = Qs[0]
        for i in range(1,time_steps):
            Qs_cum[i] = Qs_cum[i-1] @ Qs[i]
        # Qs_cum = Qs_cum.to(torch.float32)
        self.register_buffer("Qs_cumprod", Qs_cum)

    # ============= 工具函数: get Q^(t) 形状 = [B,C,C] or [C,C] ==================

    def _get_Q_t(self, t_0:torch.Tensor) -> torch.Tensor:
        """
        从 self.Qs 里获取 Q^(t_0+1).
        若 t_0.dim()==0 => 返回 [C,C].
        若 t_0.dim()==1 => 返回 [B,C,C], 其中 B=len(t_0).
        注意: t_0>=0 => in [0..T-1].
        """
        if t_0.dim()==0:
            # scalar
            return self.Qs[t_0.item()]   # => [C,C]
        else:
            # shape=[B]
            # => [B,C,C], 需要 advanced indexing
            return self.Qs[t_0]         # => [B,C,C]

    def _get_Q_cumprod(self, t_0:torch.Tensor) -> torch.Tensor:
        """
        返回 \prod_{u=1}^{t_0+1} Q^(u).
        若 t_0<0 => identity.
        区分 scalar vs vector.
        """
        if t_0.dim()==0:
            val = t_0.item()
            if val<0:
                return torch.eye(self.num_classes,device=self.Qs.device)
            return self.Qs_cumprod[val]  # => [C,C]
        else:
            # shape=[B], batch不同行
            B = t_0.shape[0]
            out = []
            for i in range(B):
                idx = t_0[i].item()
                if idx<0:
                    out.append(torch.eye(self.num_classes,device=self.Qs.device))
                else:
                    out.append(self.Qs_cumprod[idx])
            return torch.stack(out,dim=0) # => [B,C,C]

    # ============= 核心函数: q_xt_given_xtm1, etc. ====================

    def q_xt_given_xtm1(self, xtm1:torch.Tensor, t:torch.Tensor)->OneHotCategoricalBCHW:
        r"""
        前向: q(x_t|x_{t-1}; t).
        若 t是标量 => Q_t=[C,C], 否则 Q_t=[B,C,C].

        数学:
          x_t = x_{t-1} * Q^(t).
        """
        # 1) t_0= t-1
        t_0 = t-1
        # 2) 获取 Q_t
        Q_t = self._get_Q_t(t_0) # => [C,C] or [B,C,C]

        B,C,H,W = xtm1.shape
        # 3) reshape xtm1 => [B,H*W,C]
        xt_3d = xtm1.permute(0,2,3,1).reshape(B*H*W,C)

        if Q_t.dim()==2:
            # [C,C], => normal matmul
            out_2d = xt_3d @ Q_t
            out_4d = out_2d.view(B,H,W,C).permute(0,3,1,2)
        else:
            # [B,C,C], => need batched matmul
            # => we must reshape xt_3d => [B,H*W,C], but bmm requires [B,m,n] * [B,n,p]
            xt_3d_batched = xtm1.permute(0,2,3,1).reshape(B,H*W,C)  # [B,H*W,C]
            out_b = torch.bmm(xt_3d_batched, Q_t) # => [B,H*W,C]
            out_4d = out_b.view(B,H,W,C).permute(0,3,1,2)

        return OneHotCategoricalBCHW(out_4d)

    def get_Q_product(self, start:int, end:int)->torch.Tensor:
        """
        如果只需要 scalar case, 就可以 directly: self.Qs_cumprod[end]
        这里简单写: only support (start=0, end>=0) in scalar form
        """
        if end<start:
            return torch.eye(self.num_classes,device=self.Qs.device)
        if start!=0:
            raise NotImplementedError("only support get_Q_product(0,end).")

        return self.Qs_cumprod[end]

    def q_xt_given_x0(self, x0:torch.Tensor, t:torch.Tensor)->OneHotCategoricalBCHW:
        r"""
        前向: q(x_t|x_0) = x_0 * \prod_{u=1}^t Q^(u).

        若 t是标量 => shape=[C,C].
        若 t是向量 => shape=[B,C,C], 需要做 batched matmul.
        """
        t_0 = t-1
        # (t_0<0 => t=0 => x_t=x_0)
        # get prefix
        Q_prod = self._get_Q_cumprod(t_0) # => [C,C] or [B,C,C]

        B,C,H,W = x0.shape
        x0_3d = x0.permute(0,2,3,1).reshape(B,H*W,C).float()

        if Q_prod.dim()==2:
            # scalar
            out_b = x0_3d @ Q_prod
            out_4d = out_b.view(B,H,W,C).permute(0,3,1,2)
        else:
            # [B,C,C], do bmm
            out_b = torch.bmm(x0_3d, Q_prod)  # => [B,H*W,C]
            out_4d = out_b.view(B,H,W,C).permute(0,3,1,2)
        return OneHotCategoricalBCHW(out_4d)

    def theta_post(self, xt:torch.Tensor, x0:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        r"""
        后验: q(x_{t-1}|x_t,x_0) \propto q(x_t|x_{t-1}) * q(x_{t-1}|x_0).

        数学:
          A = xt @ Q^(t)^T
          B = x0 @ \prod_{u=1}^{t-1}Q^(u)
          post = normalize(A*B)
        """
        t_0 = t-1
        # if t=1 => x_{t-1}=x_0
        # => t_0=0 => we do a quick check
        if t_0.dim()==0:
            if t_0.item()==0:
                return x0
        else:
            # maybe some sample has t_0[i]==0 => then x_{t-1}=x_0
            # => we can handle that in a per-sample manner, or do a hack => any i? ignoring for brevity
            pass

        Q_t = self._get_Q_t(t_0)  # [C,C] or [B,C,C]
        Q_prod_tm1 = self._get_Q_cumprod(t_0-1) # => [C,C] or [B,C,C]

        B,C,H,W = x0.shape

        # --- B-part: x_{t-1}|x_0
        x0_3d = x0.permute(0,2,3,1).reshape(B,H*W,C).float()
        if Q_prod_tm1.dim()==2:
            B_2d = x0_3d @ Q_prod_tm1  # => [B*H*W,C]
        else:
            B_2d = torch.bmm(x0_3d, Q_prod_tm1)
        B_4d = B_2d.view(B,H,W,C).permute(0,3,1,2)

        # --- A-part: xt @ Q_t^T
        if Q_t.dim()==2:
            Q_t_T = Q_t.t()  # => [C,C]
            xt_3d = xt.permute(0,2,3,1).reshape(B,H*W,C)
            A_2d = xt_3d @ Q_t_T
        else:
            # [B,C,C]
            Q_t_T = Q_t.transpose(-2,-1)
            xt_3d = xt.permute(0,2,3,1).reshape(B,H*W,C)
            A_2d = torch.bmm(xt_3d, Q_t_T)
        A_4d = A_2d.view(B,H,W,C).permute(0,3,1,2)

        raw = A_4d * B_4d
        denom = raw.sum(dim=1,keepdim=True) + 1e-12
        return raw/denom


    # def theta_post_prob(self, xt: Tensor, theta_x0: Tensor, t: Tensor) -> Tensor:
    #     """
    #     This is equivalent to calling theta_post with all possible values of x0
    #     from 0 to C-1 and multiplying each answer times theta_x0[:, c].

    #     This should be used when x0 is unknown and what you have is a probability
    #     distribution over x0. If x0 is one-hot encoded (i.e., only 0's and 1's),
    #     use theta_post instead.
    #     """
    #     t = t - 1

    #     alphas_t = self.alphas[t][..., None, None, None]
    #     cumalphas_tm1 = self.cumalphas[t - 1][..., None, None, None, None]
    #     alphas_t[t == 0] = 0.0
    #     cumalphas_tm1[t == 0] = 1.0

    #     # We need to evaluate theta_post for all values of x0
    #     x0 = torch.eye(self.num_classes, device=xt.device)[None, :, :, None, None]
    #     # theta_xt_xtm1.shape == [B, C, H, W]
    #     theta_xt_xtm1 = alphas_t * xt + (1 - alphas_t) / self.num_classes
    #     # theta_xtm1_x0.shape == [B, C1, C2, H, W]
    #     theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes

    #     aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
    #     # theta_xtm1_xtx0 == [B, C1, C2, H, W]
    #     theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)

    #     # theta_x0.shape = [B, C, H, W]

    #     return torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, theta_x0)

    def theta_post_prob(self,
                        xt: Tensor,
                        theta_x0: Tensor,
                        t: Tensor,
                        jump_steps: int = 1) -> Tensor:
        """
        q(x_{t-k} | x_t, x_0) with k = min(jump_steps, t).
        Batch assumes every item shares the same t (as in your code).
        """
        B, C, H, W = xt.shape
        device = xt.device

        # scalar t assumed across batch
        t0 = int(t[0].item()) if t.dim() > 0 else int(t.item())
        k = max(1, min(jump_steps, t0))              # clamp for last hop
        t_idx = t0 - 1                                # 0-based Q index
        s_idx = t0 - k                                # target time (1-based: s=t-k)

        # Build multi-step kernel Q_seq = Π_{u=s+1}^{t} Q^(u)  (each Q^(u) is 0-based index u-1)
        # Here in 0-based: product Qs[s_idx] ... Qs[t_idx]
        Q_seq = self.Qs[s_idx]                        # [C,C]
        for u in range(s_idx + 1, t_idx + 1):
            Q_seq = Q_seq @ self.Qs[u]                # [C,C]

        # A-part: q(x_t | x_s) as a function of x_s → A(x_s) ∝ xt @ Q_seq^T
        xt_2d = xt.permute(0, 2, 3, 1).reshape(B * H * W, C)
        A_2d = xt_2d @ Q_seq.t()                      # [B*H*W, C]
        A = A_2d.view(B, H, W, C).permute(0, 3, 1, 2) # [B,C,H,W]

        # B-part: q(x_s | x_0=c) for all classes c via prefix Π_{u=1}^{s} Q^(u)
        Q_pref = self._get_Q_cumprod(torch.tensor(s_idx - 1, device=device))  # [C,C] or I
        if Q_pref.dim() == 2:
            Q_pref = Q_pref.unsqueeze(0).expand(B, C, C)                       # [B,C,C]
        eye = torch.eye(C, device=device).unsqueeze(0).expand(B, C, C)         # [B,C,C]
        B_x = (Q_pref @ eye).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)  # [B,C,C,H,W]

        # Posterior over x_s, then marginalize x_0 with theta_x0
        joint = A.unsqueeze(2) * B_x                                            # [B,C,C,H,W]
        post = joint / (joint.sum(dim=1, keepdim=True) + 1e-12)                 # norm over x_s
        out = torch.einsum("bcdhw,bdhw->bchw", post, theta_x0)                  # [B,C,H,W]
        return out

def kl_div_guidance(logits, label_ref):
    background_probs = torch.ones_like(label_ref[:, 0]) * -100
    background_probs = background_probs[:, None]

    label_ref = torch.cat((background_probs, label_ref), dim=1)

    dist = th.log_softmax(logits, dim=1)
    label_dist = th.log_softmax(label_ref, dim=1)

    return F.kl_div(dist, label_dist, log_target=True)


class DenoisingModel(nn.Module):

    def __init__(self, diffusion: DiffusionModel, unet: nn.Module, guidance_scale: float, guidance_scale_weighting: str,
                 guidance_loss_fn: str, dataset_file: str, label_smoothing: float = 0.0, step_T_sample:str = "majority"):
        super().__init__()
        self.diffusion = diffusion
        self.unet = unet
        self.guidance_scale = guidance_scale
        self.guidance_scale_weighting = guidance_scale_weighting
        self.guidance_loss_fn_name = guidance_loss_fn
        self.dataset_file = dataset_file
        self.step_T_sample = step_T_sample

        self._ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if guidance_loss_fn == "CE":
            self.guidance_loss_fn = self._ce_loss_fn
        elif guidance_loss_fn == "CE-2prob":
            self.guidance_loss_fn = self._cross_entropy_2probs
        elif guidance_loss_fn == "KLDiv":
            self.guidance_loss_fn = kl_div_guidance
        else:
            raise NotImplementedError(f"unsupported: {guidance_loss_fn}")

    @property
    def num_classes(self):
        return self.diffusion.num_classes

    @property
    def time_steps(self):
        return self.diffusion.time_steps

    def forward(self, x: Tensor, condition: Tensor,
                t: Optional[Tensor] = None,
                label_ref_logits: Optional[Tensor] = None,
                validation: bool = False,
                condition_features: Optional[Tensor] = None,
                get_multiscale_predictions: Optional[bool] = False) -> Union[Tensor, dict]:

        if self.training:
            if not isinstance(t, Tensor):
                raise ValueError("'t' needs to be a Tensor at training time")
            if not isinstance(x, Tensor):
                raise ValueError("'x' needs to be a Tensor at training time")
            return self.forward_step(x, condition, t, condition_features=condition_features,
                                     get_multiscale_predictions=get_multiscale_predictions)  # returns dictionary with {"diffusion_out":tensor, ...}
        else:
            if validation:
                return self.forward_step(x, condition, t, condition_features=condition_features)
            if t is None:

                return self.forward_denoising(x, condition, label_ref_logits=label_ref_logits,
                                              condition_features=condition_features)

            return self.forward_denoising(x, condition, cast(int, t.item()), label_ref_logits,
                                          condition_features=condition_features)

    def forward_step(self, x: Tensor, condition: Tensor, t: Tensor,
                     condition_features: Optional[Tensor]=None,
                     get_multiscale_predictions=False) -> Tensor:
        ret = self.unet(x, condition, t, condition_features, get_multiscale_predictions=get_multiscale_predictions)
        return ret





    def forward_denoising(
        self,
        x: Optional[Tensor],
        condition: Tensor,
        init_t: Optional[int] = None,
        label_ref_logits: Optional[Tensor] = None,
        condition_features: Optional[Tensor] = None,
        tta: bool = False,
        scales: list[float] = [0.75, 1.0, 1.25],
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Inference sampler with optional:
        - TTA (multi-scale + flip)
        - temperature scaling
        - keep soft xt mid-chain, collapse only at last step
        """
        tta=False
        temperature = 1.01

        def _single_pass(x_in, cond_in, condition_features):
            """one denoising pass with cosine schedule."""
            if init_t is None:
                T = self.time_steps
            else:
                T = init_t

            xt = x_in.float()
            B, C, H, W = xt.shape

            if label_ref_logits is not None:
                weights = self.guidance_scale_weights(label_ref_logits)
                label_ref = label_ref_logits.argmax(dim=1)

            # cosine schedule
            t_vals = make_cosine_schedule(T=T, K=10, concentrate="late", power=1.5)

            for i, t_scalar in enumerate(t_vals):
                t_next = t_vals[i+1] if i + 1 < len(t_vals) else 0
                k = t_scalar - t_next

                t_long  = torch.full((B,), t_scalar, device=xt.device, dtype=torch.long)
                t_float = t_long.float()

                ret = self.unet(xt, cond_in, t_float, condition_features)
                
                x0pred = ret["diffusion_out"]
               

                probs = self.diffusion.theta_post_prob(xt, x0pred, t_long, k)

                if label_ref_logits is not None and self.guidance_scale > 0:
                    grads = self.guidance_fn(
                        probs,
                        label_ref if self.guidance_loss_fn_name == "CE" else label_ref_logits,
                        weights
                    )
                    probs = torch.clamp(probs - grads, min=1e-12)


                
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
                xt = probs
                # 🔑 collapse only at the last step
                # is_last = (i == len(t_vals) - 1)
                # if is_last:
                #     if self.step_T_sample is None or self.step_T_sample == "majority":
                #         xt = OneHotCategoricalBCHW(probs=probs).max_prob_sample().float()
                #     elif self.step_T_sample == "confidence":
                        
                #         xt = OneHotCategoricalBCHW(probs=probs).prob_sample().float()
                # else:
                #     xt = probs  # keep soft mid-chain

            return xt

        # B, C, H, W = x.shape
        # 1) original
        B, C, H, W = x.shape
        all_probs = []

        # 1) original
        out_orig = _single_pass(x, condition, condition_features)   # return probs (B,C,H,W)
        all_probs.append(out_orig)

        # 2) horizontal flip (mirror left↔right)
        x_hf      = torch.flip(x, dims=[3])
        cond_hf   = torch.flip(condition, dims=[3])
        condf_hf  = torch.flip(condition_features, dims=[3]) if condition_features is not None else None
        out_hf    = _single_pass(x_hf, cond_hf, condf_hf)
        out_hf    = torch.flip(out_hf, dims=[3])  # flip back
        all_probs.append(out_hf)

        # 3) rot 90° (CCW)
        # x_r90     = torch.rot90(x, k=1, dims=(2, 3))
        # cond_r90  = torch.rot90(condition, k=1, dims=(2, 3))
        # condf_r90 = torch.rot90(condition_features, k=1, dims=(2, 3)) if condition_features is not None else None
        # out_r90   = _single_pass(x_r90, cond_r90, condf_r90)
        # out_r90   = torch.rot90(out_r90, k=3, dims=(2, 3))  # rotate back
        # all_probs.append(out_r90)

        # 4) rot 270° (CW)
        # x_r270     = torch.rot90(x, k=3, dims=(2, 3))
        # cond_r270  = torch.rot90(condition, k=3, dims=(2, 3))
        # condf_r270 = torch.rot90(condition_features, k=3, dims=(2, 3)) if condition_features is not None else None
        # out_r270   = _single_pass(x_r270, cond_r270, condf_r270)
        # out_r270   = torch.rot90(out_r270, k=1, dims=(2, 3))  # rotate back
        # all_probs.append(out_r270)

        # Ensemble (probabilities)
        # probs_ensemble = torch.stack(all_probs, dim=0).mean(dim=0)
        # probs_ensemble = torch.clamp(probs_ensemble, min=1e-12)
        # probs_ensemble = probs_ensemble / (probs_ensemble.sum(dim=1, keepdim=True) + 1e-12)

        views = torch.stack(all_probs, dim=0).clamp_(1e-12, 1.0)    # [V,B,C,H,W]
        logp = torch.log(views)
        probs_ensemble = torch.exp(logp.mean(dim=0))
        probs_ensemble = probs_ensemble / (probs_ensemble.sum(dim=1, keepdim=True) + 1e-12)

        tau_val = 10
        per_class_tau = [1,1,1,1,1,1,1,tau_val,tau_val,1,1,tau_val,1]  # 13 classes
        if per_class_tau is not None:              # per_class_tau: float or [C] tensor you load from disk
            probs_ensemble = apply_per_class_temperature(probs_ensemble, per_class_tau)
        # print(condition_features.shape)
        # quit()
        # probs_refined = refine_with_patches(
        #     x, condition, condition_features,
        #     probs_ensemble, _single_pass,
        #     thresh=0.6, topk=3, stride=32, halo=32
        # )

        probs_final = probs_ensemble / (probs_ensemble.sum(dim=1, keepdim=True) + 1e-12)

        if self.step_T_sample is None or self.step_T_sample == "majority":
            xt = OneHotCategoricalBCHW(probs=probs_final).max_prob_sample().float()
        else:
            xt = OneHotCategoricalBCHW(probs=probs_final).prob_sample().float()

        return {"diffusion_out": xt}









    # def forward_denoising(self, x: Optional[Tensor], condition: Tensor, init_t: Optional[int] = None,
    #                     label_ref_logits: Optional[Tensor] = None,
    #                     condition_features: Optional[Tensor] = None) -> dict:
    #     if init_t is None:
    #         init_t = self.time_steps

    #     xt = x
    #     B, C, H, W = xt.shape

    #     if label_ref_logits is not None:
    #         weights = self.guidance_scale_weights(label_ref_logits)
    #         label_ref = label_ref_logits.argmax(dim=1)

    #     t_vals = make_cosine_schedule(T=init_t, K=10, concentrate="late", power=1.5)
    #     # e.g., concentrate="both" for symmetric, "late" for more near t=1

    #     for i, t_scalar in enumerate(t_vals):
    #         t_next = t_vals[i+1] if i + 1 < len(t_vals) else 0
    #         k = t_scalar - t_next                     # exact hop size for this step

    #         t_long  = torch.full((B,), t_scalar, device=xt.device, dtype=torch.long)
    #         t_float = t_long.float()

    #         ret = self.unet(xt, condition, t_float, condition_features)
    #         x0pred = ret["diffusion_out"]

    #         # k-step posterior (deterministic)
    #         probs = self.diffusion.theta_post_prob(xt, x0pred, t_long, k)

    #         # optional deterministic guidance
    #         if label_ref_logits is not None and self.guidance_scale > 0:
    #             grads = self.guidance_fn(
    #                 probs,
    #                 label_ref if self.guidance_loss_fn_name == "CE" else label_ref_logits,
    #                 weights
    #             )
    #             probs = torch.clamp(probs - grads, min=1e-12)

    #         # renormalize over classes
    #         probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)

    #         # keep soft along the chain; collapse only at the end
    #         is_last = (i == len(t_vals) - 1)
    #         if is_last:
    #             if self.step_T_sample is None or self.step_T_sample == "majority":
    #                 xt = OneHotCategoricalBCHW(probs=probs).max_prob_sample().float()
    #             elif self.step_T_sample == "confidence":
    #                 xt = OneHotCategoricalBCHW(probs=probs).prob_sample().float()
    #         else:
    #             xt = probs
        

    #     return {"diffusion_out": xt}


    def guidance_fn(self, x, label_ref, guidance_scale_weights):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = x_in  # think identity function

            loss = self.guidance_loss_fn(logits, label_ref)
            gradients_unscaled = th.autograd.grad(outputs=loss, inputs=x_in)[0]

            scales = guidance_scale_weights * self.guidance_scale
            gradients = gradients_unscaled * scales[:, None, :, :]
            assert len(self._check_tensor(gradients)) == 0
            return gradients

    def guidance_scale_weights(self, label_ref_logits):
        label_ref = label_ref_logits.argmax(dim=1)

        if self.guidance_scale_weighting == "binary":
            guidance_scale_weights = th.zeros_like(label_ref)
            label_ref_sm = th.softmax(label_ref_logits, dim=1)

            prob_threshold = 0.5
            for cls in range(label_ref_sm.shape[1]):
                guidance_scale_weights = th.max(guidance_scale_weights, th.where(label_ref_sm[:, cls, :, :] >= prob_threshold, 1, 0))

        elif self.guidance_scale_weighting == "maxprob":
            label_ref_sm = th.softmax(label_ref_logits, dim=1)
            guidance_scale_weights = th.max(label_ref_sm, dim=1)[0]

        elif self.guidance_scale_weighting == "entropy":
            label_ref_sm = th.softmax(label_ref_logits, dim=1)
            label_ref_sm = OneHotCategoricalBCHW.channels_last(label_ref_sm)
            entropy = Categorical(probs=label_ref_sm).entropy()
            guidance_scale_weights = []  # th.zeros_like(label_ref.argmax(dim=1))
            for img_b in range(label_ref_sm.shape[0]):
                img = entropy[img_b]
                img = img.max() + img.min() - img
                guidance_scale_weights.append(img)
            guidance_scale_weights = torch.stack(guidance_scale_weights, dim=0)

        elif self.guidance_scale_weighting == "kldiv":
            label_ref_sm = th.log_softmax(label_ref_logits, dim=1)
            uniform_dist = th.ones_like(label_ref_sm) * 1 / label_ref_sm.shape[1]

            guidance_scale_weights = th.sum(F.kl_div(label_ref_sm, uniform_dist, reduction='none'), dim=1)
        elif self.guidance_scale_weighting == "default":
            guidance_scale_weights = th.ones_like(label_ref)
        else:
            raise NotImplementedError

        return guidance_scale_weights

    def _check_tensor(self, tensor: Tensor) -> list:

        invalid_values = []

        if torch.isnan(tensor).any():
            LOGGER.error("nan found in tensor!!")
            invalid_values.append("nan")

        if torch.isinf(tensor).any():
            LOGGER.error("inf found in tensor!!")
            invalid_values.append("inf")

        if (tensor.sum(dim=1) < -1e-3).any():
            LOGGER.error("negative KL divergence in tensor!!")
            invalid_values.append("neg")

        return invalid_values

    def _cross_entropy_2probs(self, probs, label_ref_logits):
        is_cityscapes = 'cityscapes' in self.dataset_file
        label_ref = th.softmax(label_ref_logits, dim=1)
        if is_cityscapes:
            background_probs = torch.zeros_like(label_ref_logits[:, 0])
            background_probs = background_probs[:, None]
            label_ref = torch.cat((label_ref, background_probs), dim=1)
            return self._ce_loss_fn(probs, label_ref)
        elif 'voc' in self.dataset_file:
            # add 22nd class that corresponds to 'void' or unlabelled.
            unlabelled_probs = torch.zeros_like(label_ref_logits[:, 0])
            unlabelled_probs = unlabelled_probs[:, None]
            label_ref = torch.cat((label_ref, unlabelled_probs), dim=1)
            return self._ce_loss_fn(probs, label_ref)
        else:
            return self._ce_loss_fn(probs, label_ref)

def make_jump_schedule(T: int, jump: int) -> list[int]:
    if T < 1 or jump < 1:
        raise ValueError("T >= 1 and jump >= 1 required")
    vals = []
    t = T
    while t > 1:
        vals.append(t)
        t = max(1, t - jump)
    vals.append(1)              # ensure we hit t=1
    # vals is strictly decreasing, ends with 1
    return vals

def make_cosine_schedule(T: int, K: int, *, concentrate: str = "both", power: float = 1.0) -> list[int]:
    """
    Cosine-eased schedule of timesteps from T down to 1 (inclusive).

    Args:
        T: total diffusion steps (e.g., 250).
        K: number of sampling points (target; after rounding, duplicates are removed).
        concentrate: "both" | "early" | "late"
            - "both": symmetric cosine (more density at *both* ends).
            - "early": bias more density near high t (early steps).
            - "late":  bias more density near low t (late steps).
        power: curve sharpness; >1 intensifies the chosen concentration.

    Returns:
        Descending list of unique ints: [T, ..., 1]
    """
    if T < 1 or K < 1:
        raise ValueError("T >= 1 and K >= 1 required")

    s = torch.linspace(0.0, 1.0, K)                     # normalized grid
    r = 0.5 * (1.0 - torch.cos(math.pi * s))            # cosine ramp in [0,1]

    if concentrate == "early":
        r = r.pow(power)                                 # more density near start (high t)
    elif concentrate == "late":
        r = 1.0 - (1.0 - r).pow(power)                   # more density near end (low t)
    elif concentrate != "both":
        raise ValueError("concentrate must be 'both', 'early', or 'late'")

    # map ramp to timesteps (monotone decreasing T -> 1)
    t_cont = T - (T - 1) * r
    t_idx = torch.round(t_cont).to(torch.long)

    # ensure strictly descending & unique, and that we hit 1
    t_idx = torch.unique_consecutive(t_idx)              # remove adjacent duplicates
    if t_idx[-1] != 1:
        t_idx = torch.cat([t_idx, torch.tensor([1], dtype=t_idx.dtype)])

    # If rounding produced ascending order, flip to descending
    if t_idx[0] < t_idx[-1]:
        t_idx = torch.flip(t_idx, dims=[0])

    return t_idx.tolist()


def apply_per_class_temperature(probs: torch.Tensor,
                                tau: torch.Tensor | float) -> torch.Tensor:
    """
    probs: [B,C,H,W], sum over C = 1
    tau:   scalar or [C] vector (τ_c > 0)
    """
    probs = probs.clamp_min(1e-12)
    if isinstance(tau, (float, int)):
        inv_tau = 1.0 / float(tau)
        logp_scaled = probs.log() * inv_tau
        return F.softmax(logp_scaled, dim=1)
    else:
        # tau is [C]
        tau = torch.as_tensor(tau, dtype=probs.dtype, device=probs.device).clamp_min(1e-3)
        inv_tau = (1.0 / tau).view(1, -1, 1, 1)
        # softmax(log p / τ_c)  ==  normalize(p ** (1/τ_c))
        logp_scaled = probs.log() * inv_tau
        return F.softmax(logp_scaled, dim=1)
def extract_bboxes(mask, min_size=16):
    """从 hard_mask 提取 bounding boxes，mask: [H,W] bool"""
    from skimage import measure
    boxes = []
    labeled = measure.label(mask.cpu().numpy(), connectivity=2)
    for region in measure.regionprops(labeled):
        y0, x0, y1, x1 = region.bbox  # 注意顺序
        if (y1 - y0) >= min_size and (x1 - x0) >= min_size:
            boxes.append((y0, y1, x0, x1))
    return boxes






# import torch
import math
from typing import Tuple, List

def cosine_weight(h: int, w: int, halo: int) -> torch.Tensor:
    """
    生成一个 [1,1,h,w] 的2D余弦窗权重，中心=1，边缘（靠近halo外沿）逐渐变小。
    halo=0 时权重全1。
    """
    if halo <= 0:
        return torch.ones(1, 1, h, w)
    yy = torch.linspace(-1, 1, h).view(h, 1)
    xx = torch.linspace(-1, 1, w).view(1, w)
    r = torch.sqrt(xx**2 + yy**2)  # 粗略径向，简洁起见
    # 把中心(-halo..+halo)对应到高权重，边缘衰减
    # 用一个平滑余弦近似：clamp 到 [0,1] 再 cos
    k = torch.clamp(1 - r, 0, 1)
    w2d = 0.5 * (1 + torch.cos(math.pi * (1 - k)))
    return w2d.view(1, 1, h, w)

def _scale_to_feat(y: int, H: int, sh: int) -> int:
    v = int(round(y * sh / float(H)))
    return max(0, min(v, sh))

def _round_to_multiple_center(y0:int, y1:int, limit:int, m:int) -> Tuple[int,int]:
    h = y1 - y0
    if h <= 0:
        return 0, min(m, limit)
    down = max(m, (h // m) * m)
    up   = max(m, ((h + m - 1) // m) * m)
    c = (y0 + y1) // 2
    def clamp(c, hh):
        y0n = max(0, min(c - hh // 2, limit - hh))
        return y0n, y0n + hh
    cand = []
    y0u, y1u = clamp(c, up)
    cand.append((abs(up - h), y0u, y1u))
    y0d, y1d = clamp(c, down)
    cand.append((abs(down - h), y0d, y1d))
    cand.sort(key=lambda z: (z[0], z[1]))
    _, a, b = cand[0]
    return a, b

def make_box_stride_aligned(y0:int, y1:int, x0:int, x1:int, H:int, W:int, stride:int=32):
    y0, y1 = _round_to_multiple_center(y0, y1, H, stride)
    x0, x1 = _round_to_multiple_center(x0, x1, W, stride)
    return y0, y1, x0, x1

def refine_with_patches(
    x, condition, condition_features,
    probs_ensemble, _single_pass,
    thresh: float = 0.6, topk: int = 3,
    stride: int = 32, halo: int = 32,
):
    """
    - halo: 在每个选框四周增加的上下文像素数（先裁更大，再回填中间原框区域）
    - stride: 已对齐；若你想自动推断，也可传入你实现的 infer_total_stride(...) 结果
    """
    from skimage import measure

    B, C, H, W = probs_ensemble.shape
    conf_map, _ = probs_ensemble.max(dim=1)  # [B,H,W]

    # 融合累加器（先放入一份全图 baseline，确保未覆盖区域保持不变）
    num = probs_ensemble.clone()
    den = torch.ones_like(probs_ensemble[:, :1, :, :])  # [B,1,H,W]

    if condition_features is not None and condition_features.ndim == 4:
        sh, sw = int(condition_features.shape[2]), int(condition_features.shape[3])
    else:
        sh = sw = None

    def extract_bboxes_bool(mask: torch.Tensor, min_size:int=16) -> List[Tuple[int,int,int,int]]:
        boxes = []
        labeled = measure.label(mask.detach().cpu().numpy().astype('uint8'), connectivity=2)
        for region in measure.regionprops(labeled):
            y0, x0, y1, x1 = region.bbox
            if (y1 - y0) >= min_size and (x1 - x0) >= min_size:
                boxes.append((y0, y1, x0, x1))
        return boxes

    for b in range(B):
        hard_mask = (conf_map[b] < thresh)
        boxes = extract_bboxes_bool(hard_mask)
        boxes = sorted(boxes, key=lambda bb: (bb[1]-bb[0])*(bb[3]-bb[2]), reverse=True)[:topk]

        for (y0, y1, x0, x1) in boxes:
            # 1) 先对齐到 stride
            y0, y1, x0, x1 = make_box_stride_aligned(y0, y1, x0, x1, H, W, stride=stride)

            # 2) 加 halo（上下文），注意不越界
            y0h = max(0, y0 - halo); y1h = min(H, y1 + halo)
            x0h = max(0, x0 - halo); x1h = min(W, x1 + halo)
            # halo 区域也做 stride 对齐，避免内部再出现对齐问题
            y0h, y1h, x0h, x1h = make_box_stride_aligned(y0h, y1h, x0h, x1h, H, W, stride=stride)

            # 3) 裁 halo patch
            x_patch    = x[b:b+1, :, y0h:y1h, x0h:x1h]
            cond_patch = condition[b:b+1, :, y0h:y1h, x0h:x1h]

            # 4) 特征图等比例索引（用 round），并保证 >0 尺寸
            if condition_features is not None and sh is not None and sw is not None:
                y0fh, y1fh = _scale_to_feat(y0h, H, sh), _scale_to_feat(y1h, H, sh)
                x0fh, x1fh = _scale_to_feat(x0h, W, sw), _scale_to_feat(x1h, W, sw)
                if y1fh <= y0fh: y1fh = min(sh, y0fh + 1)
                if x1fh <= x0fh: x1fh = min(sw, x0fh + 1)
                cf_patch = condition_features[b:b+1, :, y0fh:y1fh, x0fh:x1fh]
            else:
                cf_patch = None

            # 5) 局部推理（带上下文）
            probs_patch_halo = _single_pass(x_patch, cond_patch, cf_patch)  # [1,C,hh,ww]

            # 6) 从 halo patch 中裁出原框区域（中心）
            y0c, y1c = (y0 - y0h), (y1 - y0h)
            x0c, x1c = (x0 - x0h), (x1 - x0h)
            probs_core = probs_patch_halo[:, :, y0c:y1c, x0c:x1c]  # [1,C,h,w]

            # 7) 权重融合（余弦窗），避免接缝
            weight = cosine_weight(probs_core.shape[2], probs_core.shape[3], halo=min(halo, max(probs_core.shape[2], probs_core.shape[3])//2)).to(probs_core.device, probs_core.dtype)
            num[b:b+1, :, y0:y1, x0:x1] = num[b:b+1, :, y0:y1, x0:x1] + probs_core * weight
            den[b:b+1, :, y0:y1, x0:x1] = den[b:b+1, :, y0:y1, x0:x1] + weight

    probs_final = num / (den + 1e-12)
    return probs_final
































if __name__ == '__main__':

    from datasets.cityscapes_config import decode_target_to_color
    from PIL import Image
    from datasets.cityscapes import encode_target
    from torchvision import transforms
    from PIL import Image
    import numpy as np


    def to_numpy(tensor):
        """Tensor to numpy, calls .cpu() if necessary"""
        with torch.no_grad():
            if tensor.device.type == 'cuda':
                tensor = tensor.cpu()
            return tensor.numpy()

    def show_tensor_with_pil_label(x, save=False, name=''):
        # x B,C,H,W
        assert len(x.shape) == 4
        b,c,h,w = x.shape

        if c == 1: # integer valued labels
            x_int = x.squeeze(1).long()
        else:  # assume onehot otherwise
            x_int = torch.argmax(x, dim=1, keepdim=True).squeeze(1).long()

        # x_int is B1HW -> x_rgb is B,H,W,3
        x_rgb = decode_target_to_color(x_int)  # BCHW
        x_rgb_numpy = to_numpy(x_rgb).astype(np.uint8)

        # only show 1st image in the batch
        x_pil = Image.fromarray(x_rgb_numpy[0])
        if save:
            # x_pil.show()
            x_pil.save(name)
        return x


    # image_pil = Image.fromarray(np.array(Image.open('2008_000026.png')))
    image_pil = Image.fromarray(np.array(Image.open('aachen_000008_000019_gtFine_labelIds.png')))
    # image_pil = Image.open('aachen_000008_000019_gtFine_labelIds.png')
    # image_pil = np.array(image_pil) + 50
    image_pil = image_pil.resize(size=(256, 128), resample=Image.NEAREST)
    image_pil.show()
    im = torch.tensor(encode_target(transforms.PILToTensor()(image_pil))).long()
    x0 = torch.nn.functional.one_hot(im, 20).permute(0,3,1,2).to("cuda")
    # x0 (B,C,H,W)
    steps = 250

    _steps = list(np.arange(1, steps, 1, dtype=int))
    _, C, H, W = x0.shape

    # for s in [0.008, 0.1, 0.2, 0.3, 0.4]:
    for s in [0.5, 0.6, 0.7, 0.8]:
        noise_params = {"s": s}
        model = DiffusionModel(schedule='cosine', time_steps=steps, num_classes=20, schedule_params=noise_params).to(
            "cuda")

        for time_step in _steps:
            t = torch.tensor([time_step], dtype=torch.long).cuda()
            xt = model.q_xt_given_x0(x0, t).sample()
            if time_step % 25 == 0:
                show_tensor_with_pil_label(xt, save=True, name=f'..\\..\\logs\\noise_schedules\\s={noise_params["s"]}_{time_step}_{H}x{W}.png')
        a = 1

