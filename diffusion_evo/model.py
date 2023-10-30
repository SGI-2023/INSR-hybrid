import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from base import BaseModel, laplace, sample_random, sample_uniform, sample_boundary, divergence
from .examples import get_examples
from .visualize import draw_signal1D, save_figure


class Heat1DModel(BaseModel):
    """heat equation with constant k"""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.k = cfg.k
        self.length = cfg.length

        self.field = self._create_network(1, 1)
        self.field_prev = self._create_network(1, 1)
        self._set_require_grads(self.field_prev, False)

    @property
    def _trainable_networks(self):
        return {"field": self.field}
    
    def _sample_in_training(self):
        return sample_random(self.sample_resolution, 1, device=self.device).requires_grad_(True) * self.length / 2

    def sample_field(self, resolution, return_samples=False):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        out = self.field(grid_samples).squeeze(-1)
        if return_samples:
            return out, grid_samples.squeeze(-1)

        return out

    def sample_field_laplacian(self, resolution):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        grid_samples.requires_grad_()
        out_field = self.field(grid_samples)

        grad_computed = laplace(out_field, grid_samples)

        grad_computed_flat = grad_computed.squeeze(-1)
        grid_samples_flat = grid_samples.squeeze(-1)

        return grad_computed_flat, grid_samples_flat

    @BaseModel._timestepping
    def initialize(self, lap_model=None):
        if not hasattr(self, "init_cond_func"):
            self.init_cond_func = get_examples(self.cfg.init_cond)
        self._initialize()

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        ref = self.init_cond_func(samples)
        out = self.field(samples)
        loss_random = F.l1_loss(out, ref)

        loss_dict = {'main': loss_random}
        return loss_dict
    
    def _vis_initialize(self):
        """visualization on tb during training"""
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=1.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)

    @BaseModel._timestepping
    def step(self, lap_model):
        """heat: dudt = (k \cdot laplace)u"""
        self.field_prev.load_state_dict(self.field.state_dict())
        self._heat(lap_model)

    @BaseModel._training_loop
    def _heat(self, lap_model):
        """forward lap_model for heat"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples)
        curr_u = self.field(samples)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        # midpoint time integrator
        w = lap_model.field(samples)
        w = divergence(w, samples)
        w0 = lap_model.field_prev(samples)
        w0 = divergence(w0, samples).detach()
        loss = torch.mean((dudt - self.k * (w + w0) / 2.) ** 2)
        loss_dict = {'main': loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution // 100, 10), 1, device=self.device) * self.length / 2
        bound_u = self.field(boundary_samples)
        bc_loss = torch.mean(bound_u ** 2) * 1.
        loss_dict.update({'bc': bc_loss})

        return loss_dict

    def write_output(self, output_folder):
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        
        values = values.detach().cpu().numpy()     
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=1.0)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_values.png")
        save_figure(fig, save_path)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npz")
        np.savez(save_path, values)



        
