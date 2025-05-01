import json
import os
import random

import numpy as np
import torch
import math


def import_data(path: str, device: str="cuda:0") -> (tuple, tuple):
    inputs_tr, outputs_tr = [],[]
    inputs_te, outputs_te = [],[]
    for file_path in os.listdir(path):
        task_in_tr = []
        task_out_tr = []
        task_in_te = []
        task_out_te = []
        f_ = os.path.join(path, file_path)

        with open(f_) as data_file:
            data = json.load(data_file)
            for pair in data["train"]:
                task_in_tr.append(torch.tensor(pair["input"], device=device))
                task_out_tr.append(torch.tensor(pair["output"], device=device))

            for pair in data["test"]:
                task_in_te.append(torch.tensor(pair["input"], device=device))
                task_out_te.append(torch.tensor(pair["output"], device=device))

        inputs_tr.append(task_in_tr)
        outputs_tr.append(task_out_tr)


        inputs_te.append(task_in_te)
        outputs_te.append(task_out_te)

    return (inputs_tr, outputs_tr) , (inputs_te, outputs_te)


def max_n_colors(inputs: list[list[torch.Tensor]], outputs: list[list[torch.Tensor]]) -> int:
    max_train = 0
    for inp, outp in zip(inputs, outputs):
        for p1, p2 in zip(inp, outp):
            m1 = p1.max()
            m2 = p2.max()
            max_m = max(m1.item(), m2.item())
            max_train = max(max_m, max_train)

    return max_train



def arc_to_nca_space(n: int , tensor: torch.Tensor, num_channels: int,gene_size : int, device : str = "cuda:0",
                     mode = "rgb", gene_location: list[int] = list[1], tensor2: torch.Tensor = None, gene_location2 : list[int] = list[2], is_invis = 1) -> torch.Tensor:
    #n = tensor.unique().shape[0]
    if mode == "rgb":
        h = tensor/n * 360
        l = 0.5
        s = 0.8
        c = (1-math.fabs(2*l -1)) * s
        m = l - c/2

        xt = c * (1 - torch.abs(((h / 60) % 2) - 1))
        ct = c * (tensor > 0)
        mt = m * (tensor > 0)

        zeros = torch.zeros_like(tensor, device=device)
        sixty  = torch.ones_like(tensor, device=device) *60
        onetwenty  = torch.ones_like(tensor, device=device) *120
        oneeighty = torch.ones_like(tensor, device=device) *180
        twofourty = torch.ones_like(tensor, device=device) *240
        threehundred = torch.ones_like(tensor, device=device) *300
        threesixty = torch.ones_like(tensor, device=device) *360


        Rprime = (
                (ct * torch.logical_and(h >= zeros, h < sixty)) +
                (xt * torch.logical_and(h >= sixty, h < onetwenty)) +
                (xt * torch.logical_and(h >= twofourty, h < threehundred)) +
                (ct * torch.logical_and(h >= threehundred, h < threesixty))
        )

        Gprime = (
                (xt * torch.logical_and(h >= zeros, h < sixty)) +
                (ct * torch.logical_and(h >= sixty, h < onetwenty)) +
                (xt * torch.logical_and(h >= onetwenty, h < oneeighty)) +
                (xt * torch.logical_and(h >= oneeighty, h < twofourty))
        )

        Bprime = (
                (xt * torch.logical_and(h >= onetwenty, h < oneeighty)) +
                (ct * torch.logical_and(h >= oneeighty, h < twofourty)) +
                (ct * torch.logical_and(h >= twofourty, h < threehundred)) +
                (xt * torch.logical_and(h >= threehundred, h < threesixty))
        )

        alpha  =  (255* (tensor > 0)) * is_invis #+ (255 * (tensor < 0)/10)
        mask = tensor > 0

        binary_tensor = ((tensor.unsqueeze(-1) >> torch.arange(gene_size, device=device)) & 1).bool()
        binary_tensor = binary_tensor.float()
        underlying = (binary_tensor.clone() * mask[:, :, None]).movedim(2, 0) * 255

        padding = torch.ones_like(tensor, device=device).tile(num_channels- 4 - gene_size, 1,1) * mask


        """for gene in gene_location:

            underlying[num_channels -5 - gene, :, :] = 1

        underlying[:num_channels -5 - gene_size, :, :] = 1

        underlying = underlying * (tensor > 0) * 255"""



        R = (Rprime + mt) * 255 * is_invis
        G = (Gprime + mt) * 255 * is_invis
        B = (Bprime + mt) *255 *is_invis

        return torch.cat((R[None,...],G[None,...],B[None,...], alpha[None,...],padding*255, underlying), dim=0)/255

def arc_to_nca_space_relative_encoding(n: int, tensor: torch.Tensor, num_channels: int, gene_size: int, device: str = "cuda:0",
                     mode="rgb", gene_location: list[int] = list[1], tensor2: torch.Tensor = None,
                     gene_location2: list[int] = list[2]) -> torch.Tensor:
    # n = tensor.unique().shape[0]
    if mode == "rgb":
        h = tensor / n * 360
        l = 0.5
        s = 0.8
        c = (1 - math.fabs(2 * l - 1)) * s
        m = l - c / 2

        xt = c * (1 - torch.abs(((h / 60) % 2) - 1))
        ct = c * (tensor > 0)
        mt = m * (tensor > 0)

        zeros = torch.zeros_like(tensor, device=device)
        sixty = torch.ones_like(tensor, device=device) * 60
        onetwenty = torch.ones_like(tensor, device=device) * 120
        oneeighty = torch.ones_like(tensor, device=device) * 180
        twofourty = torch.ones_like(tensor, device=device) * 240
        threehundred = torch.ones_like(tensor, device=device) * 300
        threesixty = torch.ones_like(tensor, device=device) * 360

        Rprime = (
                (ct * torch.logical_and(h >= zeros, h < sixty)) +
                (xt * torch.logical_and(h >= sixty, h < onetwenty)) +
                (xt * torch.logical_and(h >= twofourty, h < threehundred)) +
                (ct * torch.logical_and(h >= threehundred, h < threesixty))
        )

        Gprime = (
                (xt * torch.logical_and(h >= zeros, h < sixty)) +
                (ct * torch.logical_and(h >= sixty, h < onetwenty)) +
                (xt * torch.logical_and(h >= onetwenty, h < oneeighty)) +
                (xt * torch.logical_and(h >= oneeighty, h < twofourty))
        )

        Bprime = (
                (xt * torch.logical_and(h >= onetwenty, h < oneeighty)) +
                (ct * torch.logical_and(h >= oneeighty, h < twofourty)) +
                (ct * torch.logical_and(h >= twofourty, h < threehundred)) +
                (xt * torch.logical_and(h >= threehundred, h < threesixty))
        )

        alpha = (255 * (tensor > 0))   #+ (255 * (tensor2 > 0))

        mask1 = tensor > 0
        mask2 = tensor2 > 0

        encoding = tensor + ((n+1) * tensor2)

        binary_tensor = ((encoding.unsqueeze(-1) >> torch.arange(7, device=device)) & 1).bool()


        # Convert to dtype if needed (e.g., float)
        binary_tensor = binary_tensor.float()
        underlying = (binary_tensor.clone() * (mask1 | mask2)[:,:,None]).movedim(2,0) *255

        padding = torch.ones_like(tensor, device=device) * gene_location[0]
        padding = (padding.unsqueeze(-1) >> torch.arange(gene_size - underlying.shape[0], device=device)).float()
        padding = (padding.float() * (mask1 | mask2)[:,:,None]).movedim(2,0) *255

        padding2 = torch.ones_like(tensor, device=device).tile(num_channels-padding.shape[0] - 4 - underlying.shape[0], 1, 1).float() * 255


        R = (Rprime + mt) * 255
        G = (Gprime + mt) * 255
        B = (Bprime + mt) * 255

        return torch.cat((R[None, ...], G[None, ...], B[None, ...], alpha[None, ...],padding2 ,padding,underlying), dim=0) / 255


    if mode == "same":
        alpha = 1 *  (tensor > 0)
        underlying = torch.ones_like(tensor).tile(num_channels - 4, 1, 1) * (tensor > 0)
        top = tensor/n
        zeros = torch.zeros_like(tensor, device=device)
        return torch.cat((top[None,...], zeros[None,...], zeros[None,...], alpha[None,...], underlying), dim=0)


def nca_to_rgb_image(nca_out : torch.Tensor) -> np.ndarray:

    if len(nca_out.shape) == 3:
        nca_out = nca_out[None,...]

    image = nca_out[0,:4,...].cpu().permute(1,2,0).numpy()

    return image

def filter_problems(input_problems: list[list[torch.Tensor]], output_problems: list[list[torch.Tensor]], input_eval: list[list[torch.Tensor]], output_eval: list[list[torch.Tensor]]) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]] ,list[list[torch.Tensor]]]:
    filtered_inputs = []
    filtered_outputs = []
    filtered_output_eval = []
    filtered_input_eval = []
    for i in range(len(input_problems)):
        counter = 0
        for j in range(len(input_problems[i])):
            ix, iy = input_problems[i][j].shape[-2:]
            ox, oy = output_problems[i][j].shape[-2:]
            if ix == ox and iy == oy:
                counter += 1
        if counter == len(input_problems[i]):
            filtered_inputs.append(input_problems[i])
            filtered_outputs.append(output_problems[i])
            filtered_input_eval.append(input_eval[i])
            filtered_output_eval.append(output_eval[i])
    return filtered_inputs, filtered_outputs, filtered_input_eval, filtered_output_eval


def mix_pool(pool: torch.Tensor, genesize:int , noise = False) -> torch.Tensor:
    gene_part = pool[:, :genesize,...]
    prop_part = pool[:, genesize:, ...]
    gene_perm = torch.randperm(gene_part.shape[0])
    prop_perm = torch.randperm(prop_part.shape[0])
    gene__part = gene_part[gene_perm]
    prop__part = prop_part[prop_perm]
    return torch.cat([gene__part, prop__part], dim=1)


def pad_to_size(size: list[int], tensor) -> torch.Tensor:
    yt = (size[0] -tensor.shape[-2])//2
    xt = (size[1] -tensor.shape[-1])//2

    rem_x = size[1]- (tensor.shape[-1]+(2*xt))
    rem_y = size[0]-(tensor.shape[-2]+(2*yt))
    new_tens = torch.nn.functional.pad(tensor, (xt, xt + rem_x  , yt, yt + rem_y), mode="constant", value=-1)
    return new_tens