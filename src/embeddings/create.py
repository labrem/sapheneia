import torch
import numpy as np
# Assuming the file is now in src/embeddings/create.py
from timesfm import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_torch import revin
from timesfm.torch.util import update_running_stats


class CustomTimesFm(TimesFM_2p5_200M_torch):
    """
    A custom TimesFm v2.5 class to expose embeddings, refactored for clarity.
    """

    @torch.no_grad()
    def get_embeddings(self, inputs: list):
        """
        Processes new data and returns the final transformer embeddings.

        This method orchestrates the four main steps of embedding generation:
        1. Preprocessing and padding the raw input series.
        2. Normalizing the padded inputs using the Revin technique.
        3. Running the normalized data through the transformer model.
        4. Pooling the patch embeddings into a single vector per series.

        Args:
            inputs: A list of 1D NumPy arrays or PyTorch tensors.

        Returns:
            A tuple containing patch embeddings and the final pooled embedding.
        """
        self.model.eval()

        # Step 1: Convert raw inputs to padded tensors and masks
        padded_inputs, masks = self._preprocess_and_pad(inputs)

        # Step 2: Apply reversible instance normalization (Revin)
        normed_inputs, patched_masks = self._normalize_inputs(padded_inputs, masks)

        # Step 3: Run through the model to get contextual patch embeddings
        all_outputs, _ = self.model.forward(normed_inputs, patched_masks)
        patch_embeddings = all_outputs[1]

        # Step 4: Pool the patch embeddings into a single summary vector
        pooled_embedding = torch.mean(patch_embeddings, dim=1)

        return patch_embeddings, pooled_embedding

    def _preprocess_and_pad(self, inputs: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Handles preprocessing and padding of input time series.

        This method takes a list of time series, which can have varying lengths,
        and prepares them for batch processing. It calculates the necessary
        length for padding (a multiple of the model's patch size) and then pads
        each series with leading zeros to create uniform tensors. It also
        generates a corresponding boolean mask to indicate which values are
        padded (True) and which are real data (False).

        Args:
            inputs: A list of 1D NumPy arrays or PyTorch tensors.

        Returns:
            A tuple containing:
                - padded_inputs (torch.Tensor): A tensor of all time series
                  padded to the same length. Shape: [batch_size, max_len].
                - masks (torch.Tensor): A boolean tensor indicating the
                  location of padded values. Shape: [batch_size, max_len].
        """
        all_padded_inputs = []
        all_padded_masks = []
        p = self.model.p  # Patch size, typically 32

        # Determine the maximum length required to fit all series in multiples of p
        max_len = 0
        for ts in inputs:
            max_len = max(max_len, len(ts))
        max_len = ((max_len + p - 1) // p) * p  # Round up to the nearest multiple of p

        for ts in inputs:
            if isinstance(ts, np.ndarray):
                ts = torch.from_numpy(ts)

            # Calculate padding and create the padded tensor and mask
            pad_len = max_len - len(ts)
            padded_ts = torch.cat([torch.zeros(pad_len, dtype=torch.float32), ts.float()], dim=0)
            mask = torch.cat([torch.ones(pad_len, dtype=torch.bool), torch.zeros(len(ts), dtype=torch.bool)], dim=0)

            all_padded_inputs.append(padded_ts)
            all_padded_masks.append(mask)

        # Stack the individual series into a single batch tensor
        padded_inputs = torch.stack(all_padded_inputs).to(self.model.device)
        masks = torch.stack(all_padded_masks).to(self.model.device)

        return padded_inputs, masks

    def _normalize_inputs(self, padded_inputs: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Reversible Instance Normalization (Revin) to the input tensors.

        This method is a direct implementation of the normalization step required
        by the TimesFM model. It first reshapes the data into patches. Then, it
        iterates through each patch to calculate running statistics (mean and
        standard deviation) while ignoring masked (padded) values. Finally, it
        uses these statistics to normalize the input patches. This process helps
        stabilize the model's performance across different time series scales.

        Args:
            padded_inputs: Tensor of padded input series.
            masks: Boolean tensor indicating padded values.

        Returns:
            A tuple containing:
                - normed_inputs (torch.Tensor): The normalized input patches.
                - patched_masks (torch.Tensor): The masks reshaped to match the
                  patched input structure.
        """
        batch_size, context = padded_inputs.shape
        p = self.model.p # patch size

        # Reshape the flat tensors into patches
        patched_inputs = torch.reshape(padded_inputs, (batch_size, -1, p))
        patched_masks = torch.reshape(masks, (batch_size, -1, p))
        num_input_patches = context // p

        # Calculate running statistics (mean, std dev) for each patch
        n = torch.zeros(batch_size, device=self.model.device)
        mu = torch.zeros(batch_size, device=self.model.device)
        sigma = torch.zeros(batch_size, device=self.model.device)
        patch_mu, patch_sigma = [], []

        for i in range(num_input_patches):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        # Apply the Revin normalization
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        # Ensure padded values are zeroed out after normalization
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        return normed_inputs, patched_masks