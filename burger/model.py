import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from base import BaseModel, sample_random, sample_uniform, sample_boundary, gradient
from .examples import get_examples
from .visualize import draw_signal1D, save_figure


class Burger1DModel(BaseModel):
    """burger equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.mu = cfg.mu
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

    @BaseModel._timestepping
    def initialize(self, grad_model=None):
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
        """burger: dudt = -(vel \cdot grad)u"""
        self.field_prev.load_state_dict(self.field.state_dict())
        self._burger()

    @BaseModel._training_loop
    def _burger(self):
        """forward computation for advect"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples)
        curr_u = self.field(samples)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        # midpoint time integrator

        curr_w = gradient(curr_u, samples) 
        prev_w = gradient(prev_u, samples)

        exponential_term = 0.02*torch.exp(samples*self.mu)

        grad_term_midpoint = ( curr_u)*(curr_w)

        loss = torch.mean((dudt - 0.5*(exponential_term - grad_term_midpoint)) ** 2)
        loss_dict = {'main': loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution // 100, 10), 1, device=self.device) * self.length / 2
        bound_u = self.field(boundary_samples)
        bc_loss = torch.mean(bound_u ** 2) * 1.
        loss_dict.update({'bc': bc_loss})

        return loss_dict
    
    def _compute_groundtruth(self, samples):
        '''compute the groundtruth using method of characteristics'''
        t = self.timestep * self.dt
        values_gt = self.init_cond_func(self.vis_samples - t * self.vel)
        return values_gt

    def _vis_burger(self):
        """visualization on tb during training"""
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=10.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)

    def write_output(self, output_folder):
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        
        values = values.detach().cpu().numpy()
        '''
        if "example2" in self.cfg.init_cond:
            updated_mu = [
                -1.5 + self.dt*self.timestep*self.vel,
                -1.0 + self.dt*self.timestep*self.vel,
                -0.5 + self.dt*self.timestep*self.vel
            ]
        else:
            updated_mu = -1.5 + self.dt*self.timestep*self.vel
        '''

        #ref = get_examples(self.cfg.init_cond, mu=updated_mu)(samples)
        
        samples = samples.detach().cpu().numpy()
        #ref = ref.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=1.0)  #draw_signal1D(samples, values, y_gt=ref, y_max=1.0)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_values.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npz")
        np.savez(save_path, values)

