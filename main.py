import os
from config import Config

# create experiment config containing all hyperparameters
cfg = Config("train")

# create model
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
elif cfg.pde == "elasticity":
    from elasticity import ElasticityModel as neuralModel
elif cfg.pde == "diffusion":
    from diffusion import Heat1DModel as neuralModel
elif cfg.pde == "diffusion_evo":
    from diffusion_evo import Heat1DModel as neuralModel
    from diffusion_evo import HeatLap1DModel as neuralLapModel

    lap_cfg = Config("train")
    lap_cfg.init_cond = cfg.init_cond + '_lap'
    lap_model = neuralLapModel(lap_cfg)
else:
    raise NotImplementedError
model = neuralModel(cfg)

output_folder = os.path.join(cfg.exp_dir, "results")
os.makedirs(output_folder, exist_ok=True)

# start time integration
for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    if t == 0:
        if cfg.pde == "diffusion_evo":
            lap_model.initialize()
        model.initialize()
    else:
        if cfg.pde == "diffusion_evo":
            lap_model.step()
            model.step(lap_model)
        else:
            model.step()

    print(f"Saving timestep {t} checkpoint...")
    if cfg.pde == "diffusion_evo":
        lap_model.write_output(output_folder)
    model.write_output(output_folder)
