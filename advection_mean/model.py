import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from base import BaseModel, gradient, sample_random, sample_uniform, sample_boundary
from .examples import get_examples
from .visualize import draw_signal1D, save_figure


class Advection1DModel(BaseModel):
    """advection equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.vel = cfg.vel
        self.length = cfg.length

        self.dist = nn.PairwiseDistance(p=2)

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
    
    def sample_field_gradient(self, resolution):

        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        grid_samples.requires_grad_()
        out_field = self.field(grid_samples)

        grad_computed = gradient(out_field, grid_samples)

        dist = [-0.02, -0.01, 0.01, 0.02]
        for d in dist:
            samples_up = torch.clamp(grid_samples + d, min=-2, max=2)
            curr_u_up = self.field(samples_up)
            grad_u_up = gradient(curr_u_up, samples_up)
            grad_computed = torch.cat([grad_computed, grad_u_up], dim=-1)
        grad_computed_flat = torch.mean(grad_computed, dim=-1)
        
        """
        samples_distances = torch.cdist(grid_samples, grid_samples, p=2)
        samples_distances[samples_distances == 0] = 1e-6
        dist, indices = torch.sort(samples_distances, dim=-1)
        dist = dist[:,:5]
        indices = indices[:,:5]
        gradients_neighbours = grad_computed[indices].squeeze(-1)
        dist_norm = dist/torch.sum(dist, dim=-1).unsqueeze(-1).repeat(1,dist.shape[-1])
        dist_flipped = torch.ones(dist_norm.shape).to(dist_norm.device)-dist_norm
        dist_flipped_norm = dist_flipped/(torch.sum(dist_flipped))
        w_average_gradients = torch.sum(gradients_neighbours*dist_flipped_norm, dim=-1)
        grad_computed_flat = w_average_gradients
        """

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
        curr_u = self.field(samples)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        # midpoint time integrator
        grad_u = gradient(curr_u, samples)
        grad_list = [grad_u]
        # upsamples
        dist = [-0.02, -0.01, 0.01, 0.02]
        for d in dist:
            samples_up = torch.clamp(samples + d, min=-2, max=2)
            curr_u_up = self.field(samples_up)
            grad_u_up = gradient(curr_u_up, samples_up)
            grad_list.append(grad_u_up)
        grad_stack = torch.stack(grad_list,dim=-1)
        grad_u_avg = torch.mean(grad_stack,dim=-1)
        

        """
        samples_distances = torch.cdist(samples, samples, p=2)
        samples_distances[samples_distances == 0] += 1e-6
        dist, indices = torch.sort(samples_distances, dim=-1)
        dist = dist[:,:5]
        indices = indices[:,:5]
        gradients_neighbours = grad_u[indices].squeeze(-1)
        dist_norm = dist/torch.sum(dist, dim=-1).unsqueeze(-1).repeat(1,dist.shape[-1])
        dist_flipped = torch.ones(dist_norm.shape).to(dist_norm.device)-dist_norm
        dist_flipped_norm = dist_flipped/(torch.sum(dist_flipped))
        w_average_gradients = torch.sum(gradients_neighbours*dist_flipped_norm, dim=-1)
        grad_u_averaged = w_average_gradients
        """

        grad_u0 = gradient(prev_u, samples)
        loss = torch.mean((dudt + self.vel * (grad_u_avg + grad_u0) / 2.) ** 2)
        loss_dict = {'main': loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution // 100, 10), 1, device=self.device) * self.length / 2
        bound_u = self.field(boundary_samples)
        bc_loss = torch.mean(bound_u ** 2) * 1. 
        loss_dict.update({'bc': bc_loss})

        values, samples = self.sample_field(self.vis_resolution, return_samples=True)
        values = values.detach().cpu()
        ref = get_examples(self.cfg.init_cond, mu=-1.5 + self.dt*self.t*self.vel)(samples)
        ref = ref.detach().cpu()

        mae = nn.L1Loss()
        true_error = mae(values, ref)
        loss_dict.update({'true_error': true_error})

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
        ref = get_examples(self.cfg.init_cond, mu=-1.5 + self.dt*self.t*self.vel)(samples)
        samples = samples.detach().cpu().numpy()
        ref = ref.detach().cpu().numpy()
        fig = draw_signal1D(samples, values, y_gt=ref, y_max=1.0)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_values.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npz")
        np.savez(save_path, values)


        grad_u, samples = self.sample_field_gradient(self.vis_resolution)
        grad_u_detach = grad_u.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig_grad = draw_signal1D(samples, grad_u_detach)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_nolimit.png")
        save_figure(fig_grad, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_nolimit.npz")
        np.savez(save_path, grad_u_detach)

        fig_grad = draw_signal1D(samples, grad_u_detach, y_max=1.0)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.png")
        save_figure(fig_grad, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.npz")
        np.savez(save_path, grad_u_detach)
