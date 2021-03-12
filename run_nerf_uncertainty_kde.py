import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import cv2

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

from collections import OrderedDict

from scipy import stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous().to(device)
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    variances = []
    disps = []

    t = time.time()
    if render_poses.ndim == 3:
        for i, c2w in enumerate(tqdm(render_poses)):
            print(i, time.time() - t)
            t = time.time()
            rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

            # negative to positive
            variance = torch.log(1 + torch.exp(var)) + 1e-06

            rgbs.append(rgb.cpu().numpy())
            variances.append(variance.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if i==0:
                print(rgb.shape, disp.shape)

            """
            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                print(p)
            """

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
    
    else:
        c2w = render_poses[:3,:4]
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF_KDE(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF_KDE(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        model_fine = nn.DataParallel(model_fine).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # optimizer_unc.load_state_dict(ckpt['optimizer_unc_state_dict'])
        # Load model
        pretrained_dict = ckpt['network_fn_state_dict']
        for k,v in pretrained_dict.items():
            print('loaded weights at ', k)
        model_dict = model.state_dict()
        for k,v in model_dict.items():
            print('net weights at ', k)
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # show loaded dict
        for k,v in pretrained_dict.items():
            print('loaded weights at ', k)

        ##
        pretrained_dict = ckpt['network_fine_state_dict']
        model_fine_dict = model_fine.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_fine_dict}
        # 2. overwrite entries in the existing state dict
        model_fine_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model_fine.load_state_dict(model_fine_dict)
        
        # load model_unc
        # pretrained_dict = ckpt['network_uncertainty_state_dict']
        # for k,v in pretrained_dict.items():
        #     print('loaded weights at ', k)
        # model_uncertainty_dict = model_uncertainty.state_dict()
        # for k,v in model_uncertainty_dict.items():
        #     print('net weights at ', k)
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_uncertainty_dict}
        # # 2. overwrite entries in the existing state dict
        # model_uncertainty_dict.update(pretrained_dict) 
        # # 3. load the new state dict
        # model_uncertainty.load_state_dict(model_uncertainty_dict)
        # # show loaded dict
        # for k,v in pretrained_dict.items():
        #     print('loaded weights at ', k)

    else:
        print('No reloading')

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, coarse=True):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 8]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    ## use sigmoid as activation for rgb mean output
    rgb_mean = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    ## use relu as activation for alpha mean output
    alpha_mean = torch.relu(raw[...,6])  # [N_rays, N_samples]
    ## use softplus as activation for variance output
    rgb_var = torch.log(1 + torch.exp(raw[...,3:6])) + 1e-06 # [N_rays, N_samples, 3]
    alpha_var = torch.log(1 + torch.exp(raw[...,7])) + 1e-06 # [N_rays, N_samples]

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,-1].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            # np.random.seed(0)
            noise = np.random.rand(*list(raw[...,-1].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    ## generate alpha composited color weights for importance sampling  
    alpha = 1.-torch.exp(-alpha_mean*dists) # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]
    ## sample K1 value for rgb and alpha respectively at each sampling point, 
    ## then generate K3 predictive values for each ray color
    k3 = 100
    rgb_xi = np.broadcast_to(np.random.randn(k3), list(rgb_mean.shape) + [k3]) # [N_rays, N_samples, 3, k3]
    rgb_xi = torch.tensor(rgb_xi.copy())
    # rgb_xi = torch.zeros(list(rgb_mean.shape) + [k3])
    rgbs = rgb_mean[...,None].repeat(1,1,1,k3) + rgb_xi * rgb_var[...,None].repeat(1,1,1,k3) # [N_rays, N_samples, 3, k3]
    alpha_xi = np.broadcast_to(np.random.randn(k3), list(alpha_mean.shape) + [k3]) # [N_rays, N_samples, k3]
    alpha_xi = torch.tensor(alpha_xi.copy())
    # alpha_xi = torch.zeros(list(alpha_mean.shape) + [k3])
    alphas = alpha_mean[...,None].repeat(1,1,k3) + alpha_xi * alpha_var[...,None].repeat(1,1,k3) # [N_rays, N_samples, k3]
    # print('alpha_mean:',alpha_mean)
    # print('alphas:',alphas)
    ## alphas cannot be negative
    # alphas = torch.relu(alphas)
    alphas = torch.exp(alphas)

    ## rgb cannot be negative
    rgbs = torch.relu(rgbs) 
    ## rgb and alpha should be between (0,1)
    # rgbs = torch.clamp(rgbs, min=0, max=1)
    # alphas = torch.clamp(alphas, min=0, max=1)
    

    alpha_com = 1.-torch.exp(-alphas * dists[...,None].repeat(1,1,k3)) # [N_rays, N_samples, k3]
    weights_com = alpha_com * torch.cumprod(torch.cat([torch.ones((alpha_com.shape[0], 1, alpha_com.shape[-1])), 1.-alpha_com + 1e-10], -2), -2)[:, :-1,:] # [N_rays, N_samples, k3]
    # assert (weights == weights_com[:,:,0]).any() == True

    ## generate RGB from alpha and rgb
    rgbs_map = torch.sum(weights_com[...,None,:] * rgbs, -3)  # [N_rays, 3, k3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map) + 1e-10, depth_map / (torch.sum(weights, -1) + 1e-10) + 1e-10)
    acc_map = torch.sum(weights_com, -2)

    # assert rgbs_map.isnan().any() == False
    # assert depth_map.isinf().any() == False
    # assert acc_map.isinf().any() == False
    # assert disp_map.isnan().any() == False

    if white_bkgd:
        rgbs_map = rgbs_map + (1.-acc_map[:,None,:])

    return rgbs_map, disp_map, acc_map, weights, depth_map



def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            # np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    ## coarse process
    raw = network_query_fn(pts, viewdirs, network_fn) # alpha_mean.shape (B,N,1)

    rgbs_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ## fine process
    rgbs_map_0, disp_map_0, acc_map_0 = rgbs_map, disp_map, acc_map

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

    raw = network_query_fn(pts, viewdirs, network_fine) # alpha_mean.shape (B,N,1)

    rgbs_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgbs_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgbs_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_unc", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*4, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=2000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000000, 
                        help='frequency of render_poses video saving')

    # emsemble setting
    parser.add_argument("--index_ensembles",   type=int, default=1, 
                        help='num of networks in ensembles')
    parser.add_argument("--index_step",   type=int, default=1, 
                        help='step of weights to load in ensembles')


    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        
        ## scan21
        i_all = list(np.arange(13,36))
        i_train = [13,17,31,36]
        i_val = np.array([i for i in i_all if i not in i_train])


        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
 
        # scene: lego
        # input = 5
        # input = 20
        # i_train = [1, 4, 5, 9, 17, 20, 26, 31, 35, 41, 46, 53, 54, 59, 61, 62, 66, 73, 83, 95]
        # input = 50
        # i_train = [0, 1, 2, 4, 5, 8, 9, 12, 15, 17, 19, 20, 22, 23, 26, 30, 31, 33, 34, 35, 36, 
        #             37, 38, 41, 42, 45, 46, 48, 49, 53, 54, 56, 58, 59, 61, 62, 66, 71, 73, 75, 
        #             77, 78, 79, 81, 83, 88, 91, 92, 98, 99]
        
        # train on a portion of 100 training views drums
        random.seed(10)
        i_train = random.sample(list(i_train),3)
        i_train = sorted(i_train)

        i_val = random.sample(list(i_val),20)
        i_val = sorted(i_val)

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # for synthetic scenes with large area white background, if the network overfits to this background at the beginning, it makes the result very bad.
    # so authers used central croped object images for first ~1000 iters, which require us to sample from a single image at a time
    # or you can just increase batch size or trying other optimizers such as radam or ranger might help.
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # poses = poses[i_train,:,:]
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # train
        rays_rgb_train = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb_train = np.reshape(rays_rgb_train, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_train = rays_rgb_train.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_train)
        # val
        rays_rgb_val = np.stack([rays_rgb[i] for i in i_val], 0) # val images only
        rays_rgb_val = np.reshape(rays_rgb_val, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_val = rays_rgb_val.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_val)

        print('done')
        i_batch_train = 0
        i_batch_val = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb_train = torch.Tensor(rays_rgb_train).to(device)
        rays_rgb_val = torch.Tensor(rays_rgb_val).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname +'_'+ str(args.index_ensembles)))

    start = start + 1
    idx = 0
    for i in trange(start, N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()

        ## MLP1 MLP2 loop on part1 data
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb_train[i_batch_train:i_batch_train+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays_train, target_s = batch[:2], batch[2]

            i_batch_train += N_rand
            if i_batch_train >= rays_rgb_train.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb_train.shape[0])
                rays_rgb_train = rays_rgb_train[rand_idx]
                i_batch_train = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays_train = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgbs, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays_train,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        ## compute mean and variance
        rgb_mean = torch.mean(rgbs,-1)
        rgb_var = torch.zeros(rgb_mean.shape[0])
        for k in range(rgbs.shape[-1]):
            rgb_var += torch.mean((rgbs[...,k] - rgb_mean)**2,-1) / rgbs.shape[-1]

        ## 
        rgb_avg = torch.mean(rgbs,-1)
        mse_train = img2mse(rgb_avg, target_s)
        psnr_train = mse2psnr(mse_train)

        ## print intermediate results
        # if i % 2000 == 0:
        #     mse_show = torch.mean((target_s - rgbs)**2, -1)
        #     print('train mse: ',mse_show[:10])
        #     print('train var: ',var[:10])

        # Negative Log Likelihood(NLL)
        kick = 1e-06
        loss_nll1 = torch.mean(0.5*torch.log(rgb_var+kick))
        loss_nll2 = torch.mean(0.5*torch.div(torch.mean((target_s - rgb_mean)**2, -1)+kick, rgb_var+kick))
        loss =  loss_nll1 + loss_nll2 + 5

        ## kernel density estimation
        # h = torch.tensor([0.5])
        # loss = -torch.mean(torch.log(torch.mean(torch.exp(torch.div((target_s[...,None].repeat(1,1,1,rgbs.shape[-1]) - rgbs)**2,-2*h*h)),(1,2))))
        
        if 'rgb0' in extras:
            # loss_0 = -torch.mean(torch.log(torch.mean(torch.exp(torch.div((target_s[...,None].repeat(1,1,1,rgbs.shape[-1]) - extras['rgb0'])**2,-2*h*h)),(1,2))))
            loss_0 = torch.mean((torch.mean(extras['rgb0'],-1)-target_s)**2)
            loss += loss_0
            rgb0_avg = torch.mean(extras['rgb0'],-1)
            mse_train0 = img2mse(rgb0_avg, target_s)
            psnr_train0 = mse2psnr(mse_train0)
            scalars_to_log['train/mse0'] = mse_train0.item()
            scalars_to_log['train/psnr0'] = psnr_train0.item()
        
        ## compute entropy using monto-carlo estimator

        scalars_to_log['train/mse'] = mse_train.item()
        scalars_to_log['train/pnsr'] = psnr_train.item()
        scalars_to_log['train/loss'] = loss.item()
        scalars_to_log['train/unc'] = torch.mean(rgb_var)

        ## optimize 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del rgbs, disp, acc, extras

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        scalars_to_log['iter_time'] = dt

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}_{:02d}.tar'.format(i, args.index_ensembles))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)


        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
        
        # from nerf++
        if i % args.i_img == 0 or i == start +1:
            # for train data
            idx_t = idx % len(i_train)
            idx_train = i_train[idx_t] 
            with torch.no_grad():
                rgbs, disps = render_path(torch.Tensor(poses[idx_train]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[idx_train]) # rgbs, (N, H, W, 3, k3)

            rgbs = rgbs.squeeze()
            rgbs_train = np.mean(rgbs,-1) # (B,H,W,3)
            mse_ = (rgbs_train.reshape([H,W,3])-images[idx_train].cpu().numpy())**2
            heatmap_mse_ = cv2.applyColorMap(to8b(mse_), cv2.COLORMAP_JET)
            heatmap_mse_ = cv2.cvtColor(heatmap_mse_, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            rgb_var = np.zeros([H,W])
            for k in range(rgbs.shape[-1]):
                rgb_var += np.mean((rgbs[...,k] - rgbs_train)**2,-1) / rgbs.shape[-1]

            ## compute entropy using Monto-Carlo estimator
            # rgbs = rgbs.squeeze() # (H,W,3,k3)
            # unc_map = np.zeros(list(rgbs.shape[:-2]))
            # h = 0.5
            # for m in range(rgbs.shape[-1]):
            #     cur_rgb = rgbs[...,m]
            #     cur_rgb = np.broadcast_to(cur_rgb[...,None],(list(rgbs.shape)))
            #     unc_map += -np.log(np.mean(np.exp(-(cur_rgb - rgbs)**2 / 2*h*h),(2,3)))

            # unc_map /= rgbs.shape[-1]

            print(mse_[200,190:210,:])
            print(rgb_var[200,190:210])
            heatmap_v = cv2.applyColorMap(to8b(rgb_var.reshape([H,W,1])), cv2.COLORMAP_JET)
            heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            ## correlation R for train image
            if i % 1000 == 0 or i == start+1:
                mse_r = np.mean(mse_,-1).reshape(-1) # (N,)
                # cor_mse = np.corrcoef(mse_r, variances)
                cor_mse = stats.spearmanr(mse_r, rgb_var.reshape(-1))
                print('R for train: ',cor_mse)

            img_pred = to8b(rgbs_train.reshape([H,W,3]).transpose(2,0,1))
            img_disp_pred = to8b(disps.reshape([H,W,1]).transpose(2,0,1))
            img_gt = to8b(images[idx_train].detach().cpu().numpy()).transpose(2,0,1)

            prefix='train/'
            writer.add_image(prefix + 'rgb_gt', img_gt, i)
            writer.add_image(prefix + 'rgb_pred', img_pred, i)
            writer.add_image(prefix + 'rgb_disp_pred', img_disp_pred, i)
            writer.add_image(prefix + 'heatmap_mse_', heatmap_mse_, i)
            writer.add_image(prefix + 'heatmap_v', heatmap_v, i)

            # for val data
            idx_v = idx % len(i_val)
            idx_val = i_val[idx_v]
            with torch.no_grad():
                rgbs, disps = render_path(torch.Tensor(poses[idx_val]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[idx_val]) # rgbs, (N, H*W, 3)
            
            rgbs = rgbs.squeeze()
            rgbs_val = np.mean(rgbs,-1) # (B,H,W,3)
            mse_ = (rgbs_val.reshape([H,W,3])-images[idx_val].cpu().numpy())**2
            heatmap_mse_ = cv2.applyColorMap(to8b(mse_), cv2.COLORMAP_JET)
            heatmap_mse_ = cv2.cvtColor(heatmap_mse_, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            rgb_var = np.zeros([H,W])
            for k in range(rgbs.shape[-1]):
                rgb_var += np.mean((rgbs[...,k] - rgbs_val)**2,-1) / rgbs.shape[-1]

            ## compute entropy using Monto-Carlo estimator
            # rgbs = rgbs.squeeze() # (H,W,3,k3)
            # unc_map = np.zeros(list(rgbs.shape[:-2]))
            # h = 0.5
            # for m in range(rgbs.shape[-1]):
            #     cur_rgb = rgbs[...,m]
            #     cur_rgb = np.broadcast_to(cur_rgb[...,None],(list(rgbs.shape)))
            #     unc_map += -np.log(np.mean(np.exp(-(cur_rgb - rgbs)**2 / 2*h*h),(2,3)))

            # unc_map /= rgbs.shape[-1]

            print(mse_[200,190:210,:])
            print(rgb_var[200,190:210])
            heatmap_v = cv2.applyColorMap(to8b(rgb_var.reshape([H,W,1])), cv2.COLORMAP_JET)
            heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            ## correlation R for val image
            if i % 1000 == 0 or i == start+1:
                mse_r = np.mean(mse_,-1).reshape(-1) # (N,)
                # cor_mse = np.corrcoef(mse_r, variances)
                cor_mse = stats.spearmanr(mse_r, rgb_var.reshape(-1))
                print('R for val: ',cor_mse)

            heatmap_v = cv2.applyColorMap(to8b(rgb_var.reshape([H,W,1])), cv2.COLORMAP_JET)
            heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            img_pred = to8b(rgbs_val.reshape([H,W,3]).transpose(2,0,1))
            img_disp_pred = to8b(disps.reshape([H,W,1]).transpose(2,0,1))
            img_gt = to8b(images[idx_val].detach().cpu().numpy()).transpose(2,0,1)

            prefix='val/'
            writer.add_image(prefix + 'rgb_gt', img_gt, i)
            writer.add_image(prefix + 'rgb_pred', img_pred, i)
            writer.add_image(prefix + 'rgb_disp_pred', img_disp_pred, i)
            writer.add_image(prefix + 'heatmap_mse_', heatmap_mse_, i)
            writer.add_image(prefix + 'heatmap_v', heatmap_v, i)

            # for test data
            # idx_t = idx % len(i_test)
            # idx_test = i_test[idx_t]
            # with torch.no_grad():
            #     rgbs, variances, disps = render_path(torch.Tensor(poses[idx_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[idx_test]) # rgbs, (N, H*W, 3)
            
            # mse_ = (rgbs.reshape([H,W,3])-images[idx_test].cpu().numpy())**2
            # heatmap_mse_ = cv2.applyColorMap(to8b(mse_), cv2.COLORMAP_JET)
            # heatmap_mse_ = cv2.cvtColor(heatmap_mse_, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            # ## correlation R for test image
            # if i % 10000 == 0 or i == start+1:
            #     out_rgb = rgbs.reshape(-1,3) # 
            #     target_s = images[idx_test].reshape(-1,3).cpu().numpy() # 
            #     variances = variances.reshape(-1,)
            #     mse_r = np.mean((out_rgb - target_s) ** 2, -1) # (N,)
            #     # cor_mse = np.corrcoef(mse_r, variances)
            #     cor_mse = stats.spearmanr(mse_r, variances)
            #     print('R for test: ',cor_mse)

            # img_v_pred = to8b(variances.reshape([H,W,1]).transpose(2,0,1))
            # heatmap_v = cv2.applyColorMap(to8b(variances.reshape([H,W,1])), cv2.COLORMAP_JET)
            # heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            # img_pred = to8b(rgbs.reshape([H,W,3]).transpose(2,0,1))
            # img_disp_pred = to8b(disps.reshape([H,W,1]).transpose(2,0,1))
            # img_gt = to8b(images[idx_test].detach().cpu().numpy()).transpose(2,0,1)

            # prefix='test/'
            # writer.add_image(prefix + 'rgb_gt', img_gt, i)
            # writer.add_image(prefix + 'rgb_pred', img_pred, i)
            # writer.add_image(prefix + 'rgb_v_pred', img_v_pred, i)
            # writer.add_image(prefix + 'rgb_disp_pred', img_disp_pred, i)
            # writer.add_image(prefix + 'heatmap_mse_', heatmap_mse_, i)
            # writer.add_image(prefix + 'heatmap_v', heatmap_v, i)

            idx += 1
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  MSE: {mse_train.item()}  PSNR: {psnr_train.item()}")
            for k in scalars_to_log:
                writer.add_scalar(k, scalars_to_log[k], i)

        global_step += 1

def test():

    parser = config_parser()
    args = parser.parse_args()

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        
        ## scan21
        i_all = list(np.arange(13,36))
        i_train = [13,17,31,36]
        i_val = np.array([i for i in i_all if i not in i_train])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # scene: lego
        # input = 5
        # i_train = [1, 4, 54, 61, 73]
        # input = 20
        # i_train = [1, 4, 5, 9, 17, 20, 26, 31, 35, 41, 46, 53, 54, 59, 61, 62, 66, 73, 83, 95]
        # input = 50
        # i_train = [0, 1, 2, 4, 5, 8, 9, 12, 15, 17, 19, 20, 22, 23, 26, 30, 31, 33, 34, 35, 36, 
        #             37, 38, 41, 42, 45, 46, 48, 49, 53, 54, 56, 58, 59, 61, 62, 66, 71, 73, 75, 
        #             77, 78, 79, 81, 83, 88, 91, 92, 98, 99]
        
        # train on a portion of 100 training views drums
        random.seed(10)
        i_train = random.sample(list(i_train),3)
        i_train = sorted(i_train)

        i_val_train = random.sample(list(i_val),20)
        i_val_train = sorted(i_val_train)

        i_val_test = [] 
        for fruit in i_val:
            if fruit not in i_val_train:
                i_val_test.append(fruit)
        
        i_val_test = random.sample(list(i_val_test),20)
        print('i_val_test:',i_val_test)

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # for synthetic scenes with large area white background, if the network overfits to this background at the beginning, it makes the result very bad.
    # so authers used central croped object images for first ~1000 iters, which require us to sample from a single image at a time
    # or you can just increase batch size or trying other optimizers such as radam or ranger might help.
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 100000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname +'_'+ str(args.index_ensembles)))

    model = render_kwargs_test['network_fn']
    model_fine = render_kwargs_test['network_fine']
    step = args.index_step

    mse_all = []
    psnr_all = []
    var_all = []

    start = start + 1
    idx = 0

    # load ckpt
    # ckpt_path = os.path.join(args.basedir, args.expname, '{:06d}_{:02d}.tar'.format(step, args.index_ensembles))
    # print('Reloading from', ckpt_path)
    # ckpt = torch.load(ckpt_path)

    # # for param_tensor in model_uncertainty.state_dict():
    # #         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # model.load_state_dict(ckpt['network_fn_state_dict'])
    # model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    # model_uncertainty.load_state_dict(ckpt['network_uncertainty_state_dict'])

    t = time.time()
    #i_train = np.array(list(i_train) + list(i_val))
    for i, img_i in enumerate(tqdm(i_val)):
        print(i, time.time() - t)

        # Random from one image
        target = images[img_i]
        pose = poses[img_i, :3,:4]

        #####  Core optimization loop  #####
        with torch.no_grad():
            rgbs, disps = render_path(pose, hwf, args.chunk, render_kwargs_test) # rgbs, (N, H, W, 3, k3)

        testsavedir = os.path.join(args.basedir, args.expname, 'testset_{:06d}'.format(step))
        os.makedirs(testsavedir, exist_ok=True)

        rgbs_train = np.mean(rgbs,-1) # (B,H,W,3)
        mse_ = (rgbs_train.reshape([H,W,3])-images[img_i].cpu().numpy())**2
        mse_map = to8b(mse_)
        heatmap_mse_ = cv2.applyColorMap(mse_map, cv2.COLORMAP_JET)
        heatmap_mse_ = cv2.cvtColor(heatmap_mse_, cv2.COLOR_BGR2RGB).transpose(2,0,1)

        rgbs = rgbs.squeeze() # (H,W,3,k3)
        unc_map = np.zeros(list(rgbs.shape[:-2]))
        h = 0.5
        for m in range(rgbs.shape[-1]):
            cur_rgb = rgbs[...,m]
            cur_rgb = np.broadcast_to(cur_rgb[...,None],(list(rgbs.shape)))
            unc_map += -np.log(np.mean(np.exp(-(cur_rgb - rgbs)**2 / 2*h*h),(2,3)))

        unc_map /= rgbs.shape[-1]
        unc_map = to8b(unc_map.reshape([H,W,1]))

        # print(mse_[200,190:210,:])
        # print(unc_map[200,190:210])
        heatmap_v = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)
        heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

        ## correlation R for train image
        mse_r = np.mean(mse_,-1).reshape(-1) # (N,)
        # cor_mse = np.corrcoef(mse_r, variances)
        cor_mse = stats.spearmanr(mse_r, unc_map.reshape(-1))
        print('R for train: ',cor_mse)

        # plot histogram of mse image
        r,g,b = cv2.split(mse_map)
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        for x, c in zip([b,g,r], ["b", "g", "r"]):
            xs = np.arange(256)
            ys = cv2.calcHist([x], [0], None, [256], [0,256])
            ax.plot(xs, ys.ravel(), color=c)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.savefig(testsavedir+'/hist_mse_{:02d}.png'.format(i))
        plt.close()

        ## plot histogram of var image
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        xs = np.arange(256)
        ys = cv2.calcHist([unc_map], [0], None, [256], [0,256])
        ax.plot(xs, ys.ravel(), color=c)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.savefig(testsavedir+'/hist_var_{:02d}.png'.format(i))
        plt.close()

        ## plot heatmaps
        # heatmap_mse = cv2.applyColorMap(mse_map, cv2.COLORMAP_JET)
        # heatmap_var = cv2.applyColorMap(var_map, cv2.COLORMAP_JET)

        # filename_img = os.path.join(testsavedir, '{:03d}.png'.format(img_i))
        # filename_var = os.path.join(testsavedir, '{:03d}_var.png'.format(img_i))
        # cv2.imwrite(filename_img, heatmap_mse)
        # cv2.imwrite(filename_var, heatmap_var)
        # print('saved ',filename_img)

        ## pixel-level variance 
        # out_rgb = rgbs.reshape(-1,3) # 
        # target_s = target.reshape(-1,3).cpu().numpy() # 
        # variances = variances.reshape(-1,)
        # print('var.shape:',variances.shape)
        # output_sig_pos = np.log(1 + np.exp(variances)) + 1e-06

        ## image-level variance
        # mse_mean_img = np.mean((rgbs-target.cpu().numpy())**2)
        # psnr_mean_img = -10. * np.log(mse_mean_img) / np.log([10.])
        # psnr_mean_img = psnr_mean_img.astype(float)
        # var_mean_img = np.mean(output_sig_pos)

        # mse_all.append(mse_mean_img.item())
        # psnr_all.append(psnr_mean_img.item())
        # var_all.append(var_mean_img.item())

        # print(mse_all)
        # print(psnr_all)
        # print(var_all)

        # delet = []
        # for m in range(out_rgb.shape[0]):
        #     if np.mean(target_s[m]) == 1.:
        #         delet.append(m)

        # out_rgb = np.delete(out_rgb, delet, 0)
        # target_s = np.delete(target_s, delet, 0)
        # output_sig_pos = np.delete(output_sig_pos, delet, 0)

        # # # mse, psnr
        # mse = np.mean((out_rgb - target_s) ** 2, -1) # (N,)
        # psnr = -10. * np.log(mse) / np.log([10.]) # (N,)

        # if i == 0:
        #     mse_all = mse
        #     psnr_all = psnr
        #     var_all = output_sig_pos
        # else:
        #     mse_all = np.concatenate((mse_all, mse), axis=0)
        #     psnr_all = np.concatenate((psnr_all, psnr), axis=0)
        #     var_all = np.concatenate((var_all, output_sig_pos), axis=0)

    # print(mse_all)
    # print(psnr_all)
    # print(var_all)

    ## compute R
    # cor_mse = np.corrcoef(mse_all, var_all)
    # print(cor_mse)
    # cor_psnr = np.corrcoef(psnr_all / 500.0, var_all)
    # print(cor_psnr)

    # x = np.linspace(1, len(mse_all), num=len(mse_all))

    # import matplotlib.pyplot as plt
    # plt.plot(x, mse_all, 'bo-', label='bias mean')
    # plt.plot(x, np.array(psnr_all) / 500.0, 'ko-', label='psnr mean')
    # plt.plot(x, var_all, 'ro-', label='std variance')

    # plt.legend()
    # plt.xlabel('test view')


    # save_path = testsavedir+'/variance_mse.png'
    # plt.savefig(save_path)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
    # test()
