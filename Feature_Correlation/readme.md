# Visualize the Category-wise Target-aware Correlation (CTC)
- Step 1: Preprocess the dataset and save the statistics for computing the Category-wise Target-aware Correlation.
    ```bash
    cd visualization
    python amazon_pre_pos.py
    ```
- Step 2: Compute and plot the Category-wise Target-aware Correlation.
    ```bash
    python amazon_mul_c_pos.py
    ```


<div align=center><img src="./images/ground_truth_STC.png" alt="Category-wise Target-aware Correlation" width="40%"></div>


# Visualize the learned semantic-temporal correlation
- Step 1: Save the value of position embedding and category embedding of different models after training as `p.npy` and `c.npy`.
- Step 2: Compute the learned quadruple correlation.
    ```bash
    python temporal_correlation.py
    ```
    
<div align=center><img src="./images/TIN_STC.png" alt="Learned semantic-temporal correlation of TIN" width="40%"></div>
