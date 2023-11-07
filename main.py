import os
from config import Config
import copy

# create experiment config containing all hyperparameters
cfg = Config("train")
grad_cfg = Config("train")

# create model
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
    from advection import AdvectionGrad1DModel as gradModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
elif cfg.pde == "elasticity":
    from elasticity import ElasticityModel as neuralModel
elif cfg.pde == 'burger':
    from burger import Burger1DModel as neuralModel
    from burger import BurgerGrad1DModel as gradModel
else:
    raise NotImplementedError

grad_cfg.init_cond = cfg.init_cond +  '_grad'
grad_model = gradModel(grad_cfg)

model = neuralModel(cfg)

output_folder = os.path.join(cfg.exp_dir, "results")
os.makedirs(output_folder, exist_ok=True)

# start time integration
for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    if t == 0:
        model.initialize()
        grad_model.initialize()
    else:
        old_model_u_field = copy.deepcopy(model.field)

        model.step(grad_model)
        grad_model.step(old_model_u_field)

    grad_model.write_output(output_folder)
    model.write_output(output_folder)
