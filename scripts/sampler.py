import torch
import torch.nn.functional as F
from sequence_generation.utils.flow_utils import DirichletConditionalFlow
from sequence_generation.utils.flow_utils import expand_simplex
from sequence_generation.utils.train_utils import load_generator, load_regressor

class Sampler:
    """Classifier-guided Dirichlet FM sampler — mirrors the upstream
    structure from Stark et al., "Dirichlet Flow Matching with Applications
    to DNA Sequence Design" (2024).

    Kept intentionally close to the upstream class layout so the gradient-
    based classifier guidance path (flow-score + cls-score solve) is easy
    to cross-reference. Retained as a reference/baseline alongside the
    project's main pipeline in `scripts/guided_sampler.py`, which adds
    property + uncertainty + GIL-style mask separator on top.

    Config fields read:
      model.cls_expanded_simplex
      sampling.prior_pseudocount
      sampling.guidance_scale
      sampling.scale_cls_score
      sampling.target_class
      sampling.n_steps
      model.alpha_max / alphabet_size
    """

    def __init__(self, config):
        self.config = config

        # Load pre-trained generative model
        self.model, _, _ = load_generator(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()
        self.K = config.model.alphabet_size
        self.condflow = DirichletConditionalFlow(K=config.model.alphabet_size, alpha_spacing=0.01, alpha_max=config.model.alpha_max)

        # Load property prediction model
        self.cls_model, _, _ = load_regressor(config)

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        prior_pseudocount: float = 0.1,
        flow_temp: float = 1.0,
    ):
        """
        Returns:
          - tokens: (B,L) LongTensor in {0..K-1}  (A,C,G,T)
          - (optional) probs: (B,L,K)
        """
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, self.K, device=self.device)).sample()
        xt = x0.clone()
        eye = torch.eye(self.K).to(x0)

        t_span = torch.linspace(1, self.config.model.alpha_max, steps=self.config.sampling.n_steps, device=self.device)

        for i, (s,t) in enumerate(zip(t_span[:-1], t_span[1:])):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), prior_pseudocount)

            logits = self.model(xt_expanded, t=t[None].expand(B))
            flow_probs = F.softmax(logits / flow_temp, dim=-1)

            # classifier guidance 
            # TODO: + exploration!!
            probs_cond, cls_score = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
            flow_probs = probs_cond * self.config.sampling.guidance_scale + flow_probs * (1 - self.config.sampling.guidance_scale)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            # Euler solver
            xt = xt + flow * (t-s)

        seq_pred = torch.argmax(logits, dim=-1)
        return logits, x0, seq_pred

    def get_cls_guided_flow(self, xt, alpha, p_x0_given_xt):
        B, L, K = xt.shape
        # get the matrix of scores of the conditional probability flows for each simplex corner
        cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / xt[..., None]))  # [B, L, K, K]
        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(2)[:, :, None, :]  # [B, L, K, K] now the columns sum up to 0
        # assert torch.allclose(cond_scores_mats.sum(2), torch.zeros((B, L, K)),atol=1e-4), cond_scores_mats.sum(2)

        score = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, p_x0_given_xt)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # assert torch.allclose(score.sum(2), torch.zeros((B, L)),atol=1e-4)

        cls_score = self.get_cls_score(xt, alpha[None].expand(B))
        if self.config.sampling.scale_cls_score:
            norm_score = torch.norm(score, dim=2, keepdim=True)
            norm_cls_score = torch.norm(cls_score, dim=2, keepdim=True)
            cls_score = torch.where(norm_cls_score != 0, cls_score * norm_score / norm_cls_score, cls_score)
        guided_score = cls_score + score

        Q_mats = cond_scores_mats.clone()  # [B, L, K, K]
        Q_mats[:, :, -1, :] = torch.ones((B, L, K))  # [B, L, K, K]
        guided_score_ = guided_score.clone()  # [B, L, K]
        guided_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_y = torch.linalg.solve(Q_mats, guided_score_) # [B, L, K]
        """
        # for debugging whether these probabilities also have negative entries and are off of the simplex in other ways
        cls_score_ = cls_score.clone()  # [B, L, K]
        cls_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_xt_given_y = torch.linalg.solve(Q_mats, cls_score_)

        score_guided_ = score.clone()  # [B, L, K]
        score_guided_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_back = torch.linalg.solve(Q_mats, score_guided_)
        """
        if torch.isnan(p_x0_given_xt_y).any():
            print("Warning: there were this many nans in the probs_cond of the classifier score: ", torch.isnan(p_x0_given_xt_y).sum(), "We are setting them to 0.")
            p_x0_given_xt_y = torch.nan_to_num(p_x0_given_xt_y)
        return p_x0_given_xt_y, cls_score

    def get_cls_score(self, xt, alpha):
        with torch.enable_grad():
            xt_ = xt.clone().detach().requires_grad_(True)
            xt_.requires_grad = True
            if self.config.model.cls_expanded_simplex:
                xt_, prior_weights = expand_simplex(xt, alpha[None].expand(xt_.shape[0]), self.config.sampling.prior_pseudocount)
            cls_logits = self.cls_model(xt_, t=alpha)
            loss = torch.nn.functional.cross_entropy(cls_logits, torch.ones(len(xt), dtype=torch.long, device=xt.device) * self.config.sampling.target_class).mean()
            assert not torch.isnan(loss).any()
            cls_score = - torch.autograd.grad(loss,[xt_])[0]  # need the minus because cross entropy loss puts a minus in front of log probability.
            assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,None]
        return cls_score.detach()