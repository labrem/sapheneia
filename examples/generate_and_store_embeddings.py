"""
The only way I could get this running is with TimesFM 2.5. That requires you to go to the root
    directory and clone the git repo. You cannot use pip install git+https://github.com/google-research/timesfm.git
    That doesn't work. You have to just git clone it.

    git clone https://github.com/google-research/timesfm.git
    cd timesfm

    When you git clone at the root level, it'll actually import timesFM-2.5 and you can use it to
    create embeddings.
"""

import numpy as np
from src.embeddings.create import CustomTimesFm

def main():
    repo_id="google/timesfm-2.5-200m-pytorch"

    print("Loading model...")
    custom_tfm = CustomTimesFm.from_pretrained(repo_id)
    print("Model loaded.")

    # 1. Put your raw numbers into a standard Python list.
    aapl_prices = [
        256.69, 258.02, 257.13, 255.45, 254.63, 254.43, 255.46,
        256.87, 252.31, 254.43, 256.08, 245.5, 237.88, 238.99
    ]

    # 2. Convert the list of numbers into a NumPy array.
    #    This is the standard format for numerical data in machine learning.
    aapl_series = np.array(aapl_prices)

    # 3. Place the NumPy array inside a list to create your final "new_data".
    #    The model is designed to process a batch of series at once.
    new_data = [aapl_series]

    # Now, 'new_data' is ready to be passed to the get_embeddings() function.
    # For example:
    # patch_embeddings, pooled_embedding = custom_tfm.get_embeddings(new_data)

    print("Data is ready for the model!")
    print(f"Type of new_data: {type(new_data)}")
    print(f"Number of time series in the batch: {len(new_data)}")
    print(f"Type of the first item in the list: {type(new_data[0])}")

    # 3. Get the embeddings
    patch_embeddings, pooled_embedding = custom_tfm.get_embeddings(new_data)

    # 4. Check the output shapes
    print(f"\nShape of Patch Embeddings: {patch_embeddings.shape}")
    print(f"\nStart of Patch Embeddings: {pooled_embedding[:10]}")
    # Expected: (batch_size, num_patches, model_dims) -> (2, 32, 1280) for the sine wave

    print(f"Shape of Pooled Embedding: {pooled_embedding.shape}")
    # Expected: (batch_size, model_dims) -> (2, 1280)

    # This is the vector to store in Weaviate
    vector_to_store_in_weaviate = pooled_embedding[0].cpu().numpy()
    print(f"\nVector for the first time series has shape: {vector_to_store_in_weaviate.shape}")

if __name__ == "__main__":
    main()