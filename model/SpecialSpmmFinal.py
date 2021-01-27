# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 6:43 PM
# @Author  : liuxiyang
import torch


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, edge, edge_w, size1, size2, out_features, dim):
        # assert indices.requires_grad == False
        # assert not torch.isnan(edge).any()
        # assert not torch.isnan(edge_w).any()
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([size1, size2, out_features]))
        b = torch.sparse.sum(a, dim=dim)
        ctx.size1 = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.size2 = size2
        if dim == 0:
            ctx.indices = a._indices()[1, :]
        else:
            ctx.indices = a._indices()[0, :]
        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if torch.cuda.is_available():
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None, None


class SpecialSpmmFinal(torch.nn.Module):
    def forward(self, edge, edge_w, size1, size2, out_features, dim=1):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, size1, size2, out_features, dim)
