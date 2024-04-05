import torch
import matplotlib.pyplot as plt
from bayes_linear_regression import BayesLinearRegression

torch.manual_seed(42)
num_samples = 50
x = torch.linspace(0, 5, num_samples).view(-1, 1)

true_slope = 2.0
true_intercept = -1.0
true_line = true_slope * x + true_intercept

noise = torch.normal(mean=torch.zeros(num_samples, 1), std=0.5)
obs_y = true_line + noise

obs_x = x
obs_y = obs_y

diagonal_cov = 0.1 * torch.eye(1)
prior_cov = torch.eye(1)

bayes_lr = BayesLinearRegression(obs_x, obs_y, diagonal_cov, prior_cov)
bayes_lr()

x_pred = torch.linspace(0, 5, 100).view(-1, 1)
pred_mean, pred_cov = bayes_lr.predict(x_pred)
lower_bound = (pred_mean.squeeze() - 2 * torch.sqrt(torch.diag(pred_cov)).squeeze()).numpy()
upper_bound = (pred_mean.squeeze() + 2 * torch.sqrt(torch.diag(pred_cov)).squeeze()).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(obs_x.numpy(), obs_y.numpy(), label='Observations', color='blue')
plt.plot(x_pred.numpy(), pred_mean.numpy(), label='Mean Prediction', color='red')
plt.fill_between(x_pred.numpy().flatten(),
                 lower_bound,
                 upper_bound,
                 color='orange', alpha=0.5, label='Uncertainty (2 std)')
plt.plot(x.numpy(), true_line.numpy(), label='True Line', color='green', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Linear Regression')
plt.legend()
plt.grid(True)
plt.show()