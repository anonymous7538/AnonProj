import torch
import numpy as np


class AttnMask():
    def __init__(self, args):
        self.args = args
        self._mask = None
        self.scores_shape = None
        self.patch_T = getattr(args, "dilate_round", 2)
        self.see_future = getattr(args, "tri_see_future", 0)

    def set_t(self, T):
        if isinstance(T, torch.Tensor):
            T = [int(v) for v in T.tolist()]
        if isinstance(self.patch_T, int) or not self.patch_T==T:
            self.patch_T = T
            self._mask=None

    def get_mask(self, scores_shape, device, **kwargs):
        if self._mask is None or self.scores_shape != scores_shape:
            mask_dict = {
                "TriangularCausalMask": TriangularCausalMask,
                "BlockMask": BlockMask,
                "BandMask": BandMask,
                "RandomMask": RandomMask,
                "GlobalMask": GlobalMask,
                "DilutedMask":DilutedMask,
            }
            self._mask = mask_dict[self.args.attn_mask](scores_shape=scores_shape, device=device, factor=self.args.attn_mask_factor,patch_T_list=self.patch_T,see_future=self.see_future, **kwargs).mask

            # B, H, L, S = scores_shape
            # expanded_mask = self._mask[: , :2, :, :]
            # zero_mask = torch.zeros((B, H - 2, L, S),
            #                         dtype=torch.bool,
            #                         device=device)
            # self._mask  = torch.cat([expanded_mask, zero_mask], dim=1)


    # @property
    # def mask(self, scores_shape, device="cpu", **kwargs):
    #     if self._mask is None:
    #         self._mask = self.get_mask(scores_shape, device, **kwargs)
    #     return self._mask

    @property
    def mask(self):
        return self._mask

class TriangularCausalMask():
    def __init__(self, scores_shape, device="cpu",see_future=False, **kwargs):
        B, _, L, _ = scores_shape
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            if see_future:
                self._mask = torch.tril(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            else:
                self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu", **kwargs):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class DilutedMask():
    def __init__(self, scores_shape, factor=0.25,patch_T_list = 2, device="cpu", **kwargs):
        patch_T_list = [patch_T_list] if isinstance(patch_T_list, int) else patch_T_list
        B, H, L, S = scores_shape
        base_mask = torch.ones(L, S, dtype=torch.bool, device=device)
        combined_mask = torch.ones_like(base_mask)
        for patch_T in patch_T_list:
            if patch_T < 2:
                continue
            _mask = self.generate_diluted_diagonal(base_mask.clone(), factor, gap_size=patch_T - 1)
            combined_mask &= _mask

        self._mask = combined_mask[None, None, :, :].expand(B, H, L, S)

    def generate_diluted_diagonal(
            self,
            mask: torch.Tensor,
            factor: float,
            gap_size: int = 1,
            centered: bool = True
    ) -> torch.Tensor:

        if not mask.dtype == torch.bool:
            raise ValueError("mask must be torch.bool")
        if factor <= 0:
            raise ValueError("factor should > 0")
        if gap_size < 0 or not isinstance(gap_size, int):
            raise ValueError("gap_size should be int type and >= 0")

        H, W = mask.shape
        width = max(1, int(round(W * float(factor))))

        device = mask.device
        rows = torch.arange(H, device=device).unsqueeze(1)  # (H,1)
        cols = torch.arange(W, device=device).unsqueeze(0)  # (1,W)
        col_grid = cols.expand(H, W)  # (H,W)

        if centered:
            half = width // 2
            band = (torch.abs(rows - col_grid) <= half)  # (H,W)
            diag_col = rows  # (H,1)
            dist = torch.abs(col_grid - diag_col)  # (H,W)
        else:
            band = ((col_grid - rows) >= 0) & ((col_grid - rows) < width)
            diag_col = rows
            dist = torch.abs(col_grid - diag_col)

        period = gap_size + 1
        keep_pattern = (dist % period == 0)

        mask[band & keep_pattern] = False
        return mask

    @property
    def mask(self):
        return self._mask

class BlockMask():
    def __init__(self, scores_shape, factor=0.25, device="cpu", **kwargs):
        B, H, L, S = scores_shape
        _mask = self.generate_block_diagonal(torch.ones(L, S, dtype=torch.bool).to(device), factor)
        self._mask = _mask[None, None, :].expand(B, H, L, S)

    def generate_block_diagonal(self, matrix: torch.Tensor, block_factor: float) -> torch.Tensor:
        if block_factor <= 0:
            raise ValueError("factor should > 0")

        rows, cols = matrix.shape
        block_rows = max(1, int(rows * block_factor))
        block_cols = max(1, int(cols * block_factor))

        row_block_idx = torch.arange(rows, device=matrix.device) // block_rows  # (rows,)
        col_block_idx = torch.arange(cols, device=matrix.device) // block_cols  # (cols,)

        diag_blocks = (row_block_idx.unsqueeze(1) == col_block_idx.unsqueeze(0))  # (rows, cols)
        matrix[diag_blocks] = False

        return matrix

    @property
    def mask(self):
        return self._mask


class BandMask(DilutedMask):
    def __init__(self, scores_shape, factor=0.25, device="cpu", **kwargs):
        B, H, L, S = scores_shape
        _mask = self.generate_diluted_diagonal(torch.ones(L, S, dtype=torch.bool).to(device), factor, gap_size=0)
        self._mask = _mask[None, None, :].expand(B, H, L, S)

class RandomMask():
    def __init__(self, scores_shape, factor=0.25, device="cpu", **kwargs):
        B, H, L, S = scores_shape
        _mask = self.generate_random_fill_matrix(torch.zeros(L, S, dtype=torch.bool).to(device), factor)
        self._mask = _mask[None, None, :].expand(B, H, L, S)

    def generate_random_fill_matrix(self, matrix, factor):
        L, S = matrix.shape
        total_elements = L * S
        fill_count = max(1, int(total_elements * factor))

        flat_indices = np.random.choice(total_elements, fill_count, replace=False)

        rows = flat_indices // S
        cols = flat_indices % S

        matrix[rows, cols] = True

        return matrix

    @property
    def mask(self):
        return self._mask

class GlobalMask():
    def __init__(self, scores_shape, factor=0.25, device="cpu", **kwargs):
        B, H, L, S = scores_shape
        _mask = self.generate_filled_matrix(torch.ones(L, S, dtype=torch.bool).to(device), factor)
        self._mask = _mask[None, None, :].expand(B, H, L, S)

    def generate_filled_matrix(self, matrix, factor):
        L, S = matrix.shape

        fill_rows = int(L * factor)
        fill_cols = int(S * factor)

        matrix[:fill_rows, :fill_cols] = False

        return matrix

    @property
    def mask(self):
        return self._mask
