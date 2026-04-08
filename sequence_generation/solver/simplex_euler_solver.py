import torch
from sequence_generation.solver.solver import Solver
from sequence_generation.utils.flow_utils import expand_simplex, DirichletConditionalFlow

class SimplexEulerSolver(Solver):
    """Linear solver implementation."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

        self.condflow = DirichletConditionalFlow(K=self.config.model.alphabet_size, alpha_spacing=0.001,alpha_max=self.config.model.alpha_max)

    def sample(self,
               B: int,
               L: int,
               prior_pseudocount: float = 0.1, 
               flow_temp: float = 1.0)-> torch.Tensor:
        """
        Returns:
          - tokens: (B,L) LongTensor in {0..K-1}  (A,C,G,T)
          - (optional) probs: (B,L,K)
        """
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample()
        xt = x0.clone()
        eye = torch.eye(K).to(x0)

        t_span = torch.linspace(1, self.config.model.alpha_max, steps=self.config.sampling.n_steps)

        for i, (s,t) in enumerate(zip(t_span[:-1], t_span[1:])):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), prior_pseudocount)

            # Model predicts denoising flow, rather than velocity field
            logits_pred = self.model(xt_expanded, t=t[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits_pred / flow_temp, dim=-1)

            # classifier guidance 
            # TODO: + exploration term!!
            # probs_cond, cls_score = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
            # flow_probs = probs_cond * self.config.sampling.guidance_scale + flow_probs * (1 - self.config.sampling.guidance_scale)

            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            # ODE solver as Euler solver
            xt = xt + flow * (t-s)

        seq_pred = torch.argmax(logits_pred, dim=-1)
        return logits_pred, x0, seq_pred

    def get_cls_guided_flow(self, xt, alpha, p_x0_given_xt):
        B, L, K = xt.shape
        # get the matrix of scores of the conditional probability flows for each simplex corner
        cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / xt[..., None]))  # [B, L, K, K]
        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(2)[:, :, None, :]  # [B, L, K, K] now the columns sum up to 0
        # assert torch.allclose(cond_scores_mats.sum(2), torch.zeros((B, L, K)),atol=1e-4), cond_scores_mats.sum(2)

        score = torch.einsum('ijkl,ijl->ijk', cond_scores_mats, p_x0_given_xt)  # [B, L, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # assert torch.allclose(score.sum(2), torch.zeros((B, L)),atol=1e-4)

        cls_score = self.get_cls_score(xt, alpha[None].expand(B))
        if self.config.scale_cls_score:
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
            if self.config.get("cls_expanded_simplex", None):
                xt_, prior_weights = expand_simplex(xt, alpha[None].expand(xt_.shape[0]), self.config.prior_pseudocount)
            if self.config.get("analytic_cls_score", None):
                assert self.config.dataset_type == 'toy_fixed', 'analytic_cls_score can only be calculated for fixed dataset'
                B, L, K = xt.shape

                x0_given_y = self.toy_data.data_class1.to(self.device) # num_seq, L
                x0_given_y_expanded = x0_given_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1,-1) # B, num_seq, L, 1
                xt_expanded = xt_.unsqueeze(1).expand(-1,x0_given_y_expanded.shape[1], -1, -1) # B, num_seq, L, K
                selected_xt = torch.gather(xt_expanded, dim=3, index=x0_given_y_expanded).squeeze() # [B, num_seq, L] where the indices of cls1_data were used to select entries in the K dimension
                p_xt_given_x0_y = selected_xt ** (alpha[:,None,None] - 1) # [B, num_seq, L]
                p_xt_given_y = p_xt_given_x0_y.mean(1)  # [B, L] because the probs for each x0 are the same

                x0_all = torch.cat([self.toy_data.data_class1, self.toy_data.data_class2], dim= 0).to(self.device)  # num_seq * 2, L
                x0_expanded = x0_all.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)  # B, num_seq*2, L, 1
                xt_expanded = xt_.unsqueeze(1).expand(-1, x0_expanded.shape[1], -1, -1)  # B, num_seq*2, L, K
                selected_xt = torch.gather(xt_expanded, dim=3,index=x0_expanded).squeeze()  # [B, num_seq, L] where the indices of cls1_data were used to select entries in the K dimension
                p_xt_given_x0 = selected_xt ** (alpha[:, None, None] - 1)  # [B, num_seq, L]
                p_xt = p_xt_given_x0.mean(1)  # [B, L] because the probs for each x0 are the same

                p_y_given_xt = p_xt_given_y / 2 / p_xt # [B,L] divide by 2 becaue that is p(y) (but it does not really matter because it is just a constant)
                p_y_given_xt = p_y_given_xt.prod(-1) # per sequence probabilities. Works because positions are independent.
                log_prob = torch.log(p_y_given_xt).sum()
                cls_score = torch.autograd.grad(log_prob, [xt_])[0]
            else:
                cls_logits = self.cls_model(xt_, t=alpha)
                loss = torch.nn.functional.cross_entropy(cls_logits, torch.ones(len(xt), dtype=torch.long, device=xt.device) * self.config.target_class).mean()
                assert not torch.isnan(loss).any()
                cls_score = - torch.autograd.grad(loss,[xt_])[0]  # need the minus because cross entropy loss puts a minus in front of log probability.
                assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,None]
        return cls_score.detach()