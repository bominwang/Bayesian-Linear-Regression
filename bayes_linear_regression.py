import torch


class BayesLinearRegression(object):
    def __init__(self, obs_x: torch.Tensor, obs_y: torch.Tensor,
                 diagonal_cov: torch.Tensor,
                 prior_cov: torch.Tensor):
        """
        :param obs_x: 观测输入 N × P
        :param obs_y: 观测输出 N × 1
        :param diagonal_cov: 高斯噪声协方差 σ*I
        :param prior_cov: 权重先验协方差
        """
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.diagonal_cov = diagonal_cov
        self.prior_cov = prior_cov

        self.weight_mu = None
        self.weight_cov = None

        self.pred_mean = None
        self.pred_cov = None

    def __call__(self):
        inverse_prior_cov = torch.inverse(self.prior_cov)
        self.weight_cov = torch.pow(self.diagonal_cov, -2) * self.obs_x.T @ self.obs_x + inverse_prior_cov
        self.weight_cov = torch.inverse(self.weight_cov)
        self.weight_mu = torch.pow(self.diagonal_cov, -2) * self.weight_cov @ self.obs_x.T @ self.obs_y

    def predict(self, x: torch.Tensor):
        """
        :param x: 插值点 M × P
        :return: 预测均值与方差
        """
        self.pred_mean = x @ self.weight_mu
        self.pred_cov = x @ self.weight_cov @ x.T
        return self.pred_mean, self.pred_cov
