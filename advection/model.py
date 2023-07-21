import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, finite_difference_gradient, sample_random, sample_uniform, sample_boundary
from .examples import get_examples
from .visualize import draw_signal1D, save_figure


class Advection1DModel(BaseModel):
    """advection equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.vel = cfg.vel
        self.length = cfg.length

        self.field = self._create_network(1, 1)
        self.field_prev = self._create_network(1, 1)
        self._set_require_grads(self.field_prev, False)

    @property
    def _trainable_networks(self):
        return {"field": self.field}
    
    def _sample_in_training(self):
        return sample_uniform(self.sample_resolution, 1, device=self.device) * self.length / 2

    def sample_field(self, resolution, return_samples=False):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        out = self.field(grid_samples).squeeze(-1)
        if return_samples:
            return out, grid_samples.squeeze(-1)
        return out
    
    def sample_field_finite_gradient(self, resolution):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        out_field = self.field(grid_samples).squeeze(-1)
        
        grid_samples_flat = grid_samples.squeeze(-1)

        grad_computed = finite_difference_gradient(out_field, grid_samples_flat)

        return grad_computed, grid_samples_flat

    @BaseModel._timestepping
    def initialize(self):
        if not hasattr(self, "init_cond_func"):
            self.init_cond_func = get_examples(self.cfg.init_cond)
        self._initialize()

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        ref = self.init_cond_func(samples)
        out = self.field(samples)
        loss_random = F.mse_loss(out, ref)

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
    def step(self):
        """advection: dudt = -(vel \cdot grad)u"""
        self.field_prev.load_state_dict(self.field.state_dict())
        self._advect()

    @BaseModel._training_loop
    def _advect(self):
        """forward computation for advect"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples).squeeze(-1)
        curr_u = self.field(samples).squeeze(-1)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        # midpoint time integrator
        grad_u = finite_difference_gradient(curr_u, samples.squeeze(-1))
        grad_u0 = finite_difference_gradient(prev_u, samples.squeeze(-1)).detach()
        loss = torch.mean((dudt + self.vel * (grad_u + grad_u0) / 2.) ** 2)
        loss_dict = {'main': loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution // 100, 10), 1, device=self.device) * self.length / 2
        bound_u = self.field(boundary_samples)
        bc_loss = torch.mean(bound_u ** 2) * 1.
        loss_dict.update({'bc': bc_loss})

        return loss_dict

    def _vis_advect(self):
        """visualization on tb during training"""
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=1.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)

    def write_output(self, output_folder):
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        
        fig = draw_signal1D(samples, values, y_min=0.0, y_max=1.0)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npz")
        np.savez(save_path, values)

        grad_u, samples = self.sample_field_finite_gradient(self.vis_resolution)
        grad_u_detach = grad_u.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()

        fig_grad = draw_signal1D(samples, grad_u_detach)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.png")
        save_figure(fig_grad, save_path)

        fig_grad_closeup = draw_signal1D(samples, grad_u_detach, y_min=-0.5, y_max=0.5)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_closeup.png")
        save_figure(fig_grad_closeup, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.npz")
        np.savez(save_path, grad_u_detach)