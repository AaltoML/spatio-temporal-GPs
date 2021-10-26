import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, ProductStructureKernel, GridInterpolationKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np
from loguru import logger
import pickle
from timeit import default_timer as timer
import matplotlib.pyplot as plt


inducing_type = 'all_time'  # 'default'
num_z = 30
likelihood_noise = 5.
kernel_lengthscales = [0.01, 0.2, 0.2]
step_size = 0.1
epochs = 20
init_params = {}
optimizer = torch.optim.Adam


# get data file names
train_name = 'train_data_0.pickle'
pred_name = 'pred_data_0.pickle'

# ===========================Load Data===========================
data = pickle.load(open(f'/Users/wilkinw1/postdoc/inprogress/Newt-dev/newt/experiments/air_quality/data/{train_name}', "rb"))
pred_data = pickle.load(open(f'/Users/wilkinw1/postdoc/inprogress/Newt-dev/newt/experiments/air_quality/data/{pred_name}', "rb" ))
train_data = data

X = data['X']
print(X.shape)

Y = np.squeeze(data['Y'])

non_nan_idx = ~np.isnan(Y)

X = X[non_nan_idx]
Y = Y[non_nan_idx]

D = X.shape[1]
Nt = 2159  # X.shape[0]

X = torch.tensor(X).float()[:2159*2]
Y = torch.tensor(Y).float()[:2159*2]
X_pred_timeseries = torch.tensor(pred_data['timeseries']['X'][:2159]).float()


class GPRegressionModelSKI(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, kernel, likelihood):
        super(GPRegressionModelSKI, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        self.base_covar_module = kernel
        self.base_covar_module.lengthscale = torch.tensor(kernel_lengthscales)
        logger.info(f'kernel_lengthscales : {kernel_lengthscales}')

        if inducing_type == 'default':
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            init_params['grid_size'] = grid_size

        elif inducing_type == 'all_time':
            grid_size = np.array([Nt, np.ceil(np.sqrt(num_z)), np.ceil(np.sqrt(num_z))]).astype(int)
            init_params['grid_size'] = grid_size

        logger.info(f'grid_size : {grid_size}')

        self.covar_module = ScaleKernel(
            GridInterpolationKernel(self.base_covar_module, grid_size, num_dims=D)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPRegressionModelSKIP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, kernel, likelihood):
        super(GPRegressionModelSKIP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if inducing_type == 'default':
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            logger.info(f'grid_size default: {grid_size}')
            init_params['grid_size'] = grid_size

        elif inducing_type == 'all_time':
            grid_size = np.array([Nt, num_z, num_z]).astype(int)
            init_params['grid_size'] = grid_size

        time_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=1, nu=1.5)
        time_kernel.lengthscale = torch.tensor([kernel_lengthscales[0]])
        space_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=1, nu=1.5)
        space_kernel.lengthscale = torch.tensor([kernel_lengthscales[1]])

        # this version breaks prediciton:
        self.covar_module = ProductStructureKernel(
            ScaleKernel(
                GridInterpolationKernel(
                    time_kernel,
                    grid_size=Nt,
                    num_dims=1
                )
            ),
            num_dims=1,
            active_dims=0
        ) * ProductStructureKernel(
            ScaleKernel(
                GridInterpolationKernel(
                    space_kernel,
                    grid_size=num_z,
                    num_dims=1
                )
            ),
            num_dims=D-1,
            active_dims=list(range(D))[1:]  # ignore first dimension
        )

        # this version works:
        # self.covar_module = ProductStructureKernel(
        #     ScaleKernel(
        #         GridInterpolationKernel(time_kernel, grid_size=100, num_dims=1)
        #     ), num_dims=D
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


kern = MaternKernel(ard_num_dims=3, nu=1.5)
# kern = MaternKernel(ard_num_dims=1, nu=1.5)
lik = gpytorch.likelihoods.GaussianLikelihood()
lik.noise = torch.tensor(likelihood_noise)

model = GPRegressionModelSKI(X, Y, kern, lik)  # SKI model
# model = GPRegressionModelSKIP(X, Y, kern, lik)  # SKIP model

# train
model.train()
lik.train()

# Use the adam optimizer
optimizer = optimizer(model.parameters(), lr=step_size)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

loss_arr = []


def train():

    for i in range(epochs):
        # Zero backprop gradients
        optimizer.zero_grad()

        with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):

            # Get output from model
            output = model(X)

            # Calc loss and backprop derivatives
            loss = -mll(output, torch.squeeze(Y))
            loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, epochs, loss.item()))

        loss_arr.append(loss.detach().numpy())

        optimizer.step()
        torch.cuda.empty_cache()


start = timer()

with gpytorch.settings.use_toeplitz(True):
    train()

end = timer()

training_time = end - start


#===========================Predict===========================

model.eval()
lik.eval()

logger.info('Predicting')

# --- SKI predictions ---

# def _prediction_fn(XS):
#     XS = torch.tensor(XS).float()
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         preds = lik(model(XS))
#         mean, var = preds.mean, preds.variance
#         mean = mean.detach().numpy()
#         var = var.detach().numpy()
#         return mean, var

# --- SKIP prediction ---

with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
        preds = model(X_pred_timeseries)


plt.plot(preds.mean)
plt.show()

