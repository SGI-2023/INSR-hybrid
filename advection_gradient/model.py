import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, gradient, sample_random, sample_uniform, sample_boundary
from .examples import get_examples
from .visualize import draw_signal1D, save_figure


class Advection1DModel(BaseModel):
    """advection equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.vel = cfg.vel
        self.length = cfg.length

        self.field = self._create_network(1, 2)
        self.field_prev = self._create_network(1, 2)
        self._set_require_grads(self.field_prev, False)

    @property
    def _trainable_networks(self):
        return {"field": self.field}
    
    def _sample_in_training(self):
        return sample_random(self.sample_resolution, 1, device=self.device).requires_grad_(True) * self.length / 2

    def sample_field(self, resolution, return_samples=False):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        out = self.field(grid_samples)[:,0].unsqueeze(-1)
        if return_samples:
            return out, grid_samples.squeeze(-1)
        

        return out
    
    def sample_field_gradient(self, resolution):

        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        grid_samples.requires_grad_()
        result = self.field(grid_samples)
        out_field = result[:,0].unsqueeze(-1)
        grad_computed_flat = result[:,1]

        # grad_computed = gradient(out_field, grid_samples)

        # grad_computed_flat = grad_computed.squeeze(-1)
        grid_samples_flat = grid_samples.squeeze(-1)

        return grad_computed_flat, grid_samples_flat


    @BaseModel._timestepping
    def initialize(self):
        if not hasattr(self, "init_cond_func"):
            self.init_cond_func = get_examples(self.cfg.init_cond)
        self._initialize()

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        samples.requires_grad_()
        ref = self.init_cond_func(samples)
        result = self.field(samples)
        out = result[:, 0].unsqueeze(-1)
        grad = result[:, 1].unsqueeze(-1)

        loss_random = F.mse_loss(out, ref)
        loss_grad = F.mse_loss(grad, gradient(ref, samples))

        loss_dict = {'main': loss_random, 'grad': loss_grad}
        
        return loss_dict
    
    def _vis_initialize(self):
        """visualization on tb during training"""
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=1.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)


        grad_u, samples = self.sample_field_gradient(self.vis_resolution)
        grad_u_detach = grad_u.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig_grad = draw_signal1D(samples, grad_u_detach, y_max=10.)
        self.tb.add_figure("field_grad", fig_grad, global_step=self.train_step)

    @BaseModel._timestepping
    def step(self):
        """advection: dudt = -(vel \cdot grad)u"""
        self.field_prev.load_state_dict(self.field.state_dict())
        self._advect()

    @BaseModel._training_loop
    def _advect(self):
        """forward computation for advect"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples)
        grad_u0 = prev_u[:, 1].unsqueeze(-1)
        prev_u = prev_u[:, 0].unsqueeze(-1)
        curr_u = self.field(samples)
        grad_u = curr_u[:, 1].unsqueeze(-1)
        curr_u = curr_u[:, 0].unsqueeze(-1)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        loss_grad = F.mse_loss(grad_u, gradient(curr_u, samples))


        loss = torch.mean((dudt + self.vel * (grad_u + grad_u0) / 2.) ** 2)
        extra_loss = 0.1*loss_grad
        loss_dict = {'main': loss, 'extra_loss': extra_loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution // 100, 10), 1, device=self.device) * self.length / 2
        bound_u = self.field(boundary_samples)[:, 0].unsqueeze(-1)
        bc_loss = torch.mean(bound_u ** 2) * 1.
        loss_dict.update({'bc': bc_loss})



        return loss_dict

    def _vis_advect(self):
        """visualization on tb during training"""
        grad_u, samples = self.sample_field_gradient(self.vis_resolution)
        grad_u_detach = grad_u.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig_grad = draw_signal1D(samples, grad_u_detach, y_max=1.0)
        self.tb.add_figure("field_grad", fig_grad, global_step=self.train_step)



        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=10.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)
        



    def write_output(self, output_folder):
        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_max=1.0)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_values.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npz")
        np.savez(save_path, values)


        grad_u, samples = self.sample_field_gradient(self.vis_resolution)
        grad_u_detach = grad_u.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig_grad = draw_signal1D(samples, grad_u_detach)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.png")
        save_figure(fig_grad, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.npz")
        np.savez(save_path, grad_u_detach)
