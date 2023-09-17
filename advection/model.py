import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from base import BaseModel, gradient, sample_random, sample_uniform, sample_boundary
from .examples import get_examples, get_examples_grad, gaussian_gradient
from .visualize import draw_signal1D, save_figure


class Advection1DModel(BaseModel):
    """advection equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.vel = cfg.vel
        self.length = cfg.length

        self.rel_w = cfg.rel_w

        self.field = self._create_network(1, 1)
        self.field_prev = self._create_network(1, 1)
        self._set_require_grads(self.field_prev, False)

        self.grad_field = self._create_network(1, 1)
        self.grad_field_prev = self._create_network(1, 1)
        self._set_require_grads(self.grad_field_prev, False)

    @property
    def _trainable_networks(self):
        return {"field": self.field, "grad_field": self.grad_field}
    
    def _sample_in_training(self):
        return sample_random(self.sample_resolution, 1, device=self.device).requires_grad_(True) * self.length / 2

    def sample_field(self, resolution, return_samples=False):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        out = self.field(grid_samples).squeeze(-1)
        if return_samples:
            return out, grid_samples.squeeze(-1)

        return out
    
    def sample_field_gradient(self, resolution, return_samples=True):

        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 1, device=self.device) * self.length / 2
        grid_samples.requires_grad_()
        out_field = self.grad_field(grid_samples).squeeze(-1)
        if return_samples:
                    return out_field, grid_samples.squeeze(-1)

        return out_field


    @BaseModel._timestepping
    def initialize(self):
        if not hasattr(self, "init_cond_func"):
            self.init_cond_func = get_examples(self.cfg.init_cond)

        if not hasattr(self, "init_cond_grad_func"):
            self.init_cond_grad_func = get_examples_grad(self.cfg.init_cond)

        self._initialize()

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        ref = self.init_cond_func(samples)
        out = self.field(samples)
        loss_random = F.mse_loss(out, ref)

        grad_ref = self.init_cond_grad_func(samples)
        grad_out = self.grad_field(out, samples)
        grad_loss_random = F.mse_loss(grad_out, grad_ref)*self.rel_w

        loss_dict = {'main': loss_random, 'grad': grad_loss_random}
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
        self.grad_field_prev.load_state_dict(self.grad_field.state_dict())
        self._advect()

    @BaseModel._training_loop
    def _advect(self):
        """forward computation for advect"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples)
        curr_u = self.field(samples)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        # midpoint time integrator
        curr_w = self.grad_field(samples)
        prev_w = self.grad_field_prev(samples)
        grad_dudt = (curr_w - prev_w) / self.dt

        grad_w = gradient(curr_w, samples)
        grad_w0 = gradient(prev_w, samples).detach()
        grad_loss = torch.mean((grad_dudt + self.vel * (grad_w + grad_w0) / 2.) ** 2)
        loss_dict = {'grad': grad_loss}

        loss = torch.mean((dudt + self.vel * (curr_w + prev_w) / 2.) ** 2)
        loss_dict.update({'main': loss})

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
        ref = get_examples(self.cfg.init_cond, mu=-1.5 + self.dt*self.timestep*self.vel)(samples)
        

        grad_ref = get_examples_grad(self.cfg.init_cond, mu=-1.5 + self.dt*self.timestep*self.vel)(samples).detach().cpu().numpy()

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
        fig_grad = draw_signal1D(samples, grad_u_detach, y_max=1.0)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.png")
        save_figure(fig_grad, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad.npz")
        np.savez(save_path, grad_u_detach)

        fig_grad = draw_signal1D(samples, grad_u_detach)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_nolimit.png")
        save_figure(fig_grad, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_nolimit.npz")
        np.savez(save_path, grad_u_detach)

        fig_grad = draw_signal1D(samples, grad_ref, y_max=1.0)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_ref.png")
        save_figure(fig_grad, save_path)

        fig_grad = draw_signal1D(samples, grad_ref)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_grad_ref_nolimit.png")
        save_figure(fig_grad, save_path)

        grad_loss = np.mean(np.abs(grad_u_detach - grad_ref))
        print("Grad Loss: ", grad_loss)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_loss.txt")
        with open(save_path, "w") as f:
            f.write(str(grad_loss))
        
