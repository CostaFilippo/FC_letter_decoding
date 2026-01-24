import numpy as np
from copy import deepcopy

def compute_session_alignment(data, condition_vars, pars):
    """
    Performs the session alignment across multiple experiments/sessions.

    Parameters
    ----------
    data : list
        A list (length = n_experiments) of dictionaries. Each dictionary 
        contains single-trial data for one experiment, including:
          - data[i]["response"] : np.array of shape (n_units, n_timepoints, n_trials)
          - possibly other fields needed for alignment.
    condition_vars : list
        A list specifying which task variables define the conditions 
        used to compute alignment. Each element is passed to `sort_trials_by_condition`.
    pars : dict
        Parameter dictionary containing, at a minimum:
          - pars["alignment"]["nPCs_align"]: how many PCs to keep
          - pars["align_proj"]["condition_vars"]: condition variables for the final
            trial-averaged projection.
          - other fields as needed.

    Returns
    -------
    D_align : list
        A list of length n_experiments. Each item is a dictionary similar to the 
        input data dictionary but with the "response" field replaced by the aligned
        single-trial response in PCA space (nPCs x time x trials).
    D_cond_avg_proj_align : np.ndarray
        The trial-averaged aligned data for each session, concatenated. The shape 
        will typically be (nPCs, n_timepoints, n_conditions, n_sessions), or some 
        similar arrangement depending on your data.
    task_conds_proj : object
        Whatever `sort_trials_by_condition` returns for the final pass 
        (the condition definitions). Typically a table-like or DataFrame-like structure.
    align_stats : dict
        Contains:
          - "varExp_pc": cumulative variance explained across the global PCA
          - "varExp_ses": a list of per-session variance explained
          - "align_sim": alignment similarity measure (median correlation across sessions)
    """
    assert isinstance(data, list), "data must be a list of session dictionaries"
    assert isinstance(condition_vars, list), "condition_vars must be a list"

    num_sessions = len(data)
    # Prepare output containers
    D_align = [None] * num_sessions
    varExp_ses = [None] * num_sessions

    # ----------------------------------------------------
    # 1) Sort each session's data by condition, 
    #    then compute trial means to build a big matrix D_avg
    # ----------------------------------------------------
    row_idxs_all = [None] * num_sessions
    row_count = 0

    # We'll accumulate all trial-averaged means into a single 3D array: 
    #   shape = [row_count, n_time, n_conditions?]
    # but that depends on how many conditions are across sessions.
    # 
    # In the MATLAB code, D_avg is a 3D array where:
    #   first dim = "units across all sessions"
    #   second dim = "time"
    #   third dim = "conditions"
    # 
    # However, in your MATLAB code, it appears you assume the 
    # same number of conditions for each session, or you 
    # flatten them. We'll replicate the logic carefully.
    D_cond_all = []

    # We need to store how many units each session has, 
    # so we can define row_idxs for each session
    all_session_means = []  # list of shape [n_units, n_time, n_conditions_for_session]
    for i_session in range(num_sessions):
        # 1a) group data by condition
        D_cond = sort_trials_by_condition(data[i_session], condition_vars)
        # D_cond is a list of 'condition_count' dictionaries, 
        # each with shape [n_units, n_timepoints, n_trials]

        num_units = D_cond[0]["response"].shape[0]
        row_idxs = np.arange(row_count, row_count + num_units)
        row_idxs_all[i_session] = row_idxs
        row_count += num_units

        # 1b) compute mean across the trial dimension for each condition
        # so shape => [n_units, n_timepoints] for each condition
        means_this_session = []
        for cdict in D_cond:
            resp_mean = np.mean(cdict["response"], axis=2)  # shape (n_units, n_time)
            means_this_session.append(resp_mean)
        #print(np.array(means_this_session).shape)
        # Stack across conditions => shape (n_units, n_time, n_conditions)
        #print(np.array(means_this_session[0]).shape,np.array(means_this_session[1]).shape)
        means_this_session = np.stack(means_this_session, axis=2)

        all_session_means.append(means_this_session)

    # Next, we combine these into one big array: D_avg
    #   dimension 0 = "units across all sessions"
    #   dimension 1 = time
    #   dimension 2 = conditions
    # But the code presupposes the same number of conditions across sessions 
    # (the code used cat(3,...) in MATLAB for each session block).
    # 
    # So let's check how many conditions:
    n_conditions = all_session_means[0].shape[2]
    # build the final D_avg with shape = [row_count, n_time, n_conditions]
    # to match the MATLAB indexing:
    n_time = all_session_means[0].shape[1]
    D_avg = np.zeros((row_count, n_time, n_conditions), dtype=np.float64)

    # Now fill it in
    row_cursor = 0
    for i_session in range(num_sessions):
        this_block = all_session_means[i_session]  # shape [n_units, n_time, n_cond]
        n_units = this_block.shape[0]
        # fill slice
        D_avg[row_cursor: row_cursor + n_units, :, :] = this_block
        row_cursor += n_units

    # ----------------------------------------------------
    # 2) Perform PCA on the aggregated data
    #    The code does: [coeff,~,~,~,explained,mu] = pca(D_avg')
    #    That means we flatten the last two dims to do PCA on shape => [row_count, n_time * n_cond].
    #    Then we transpose => shape => [n_time * n_cond, row_count].
    # ----------------------------------------------------
    # Flatten across time and conditions
    D_avg_2d = D_avg.reshape(row_count, -1)  # shape [row_count, n_time*n_cond]
    # Then transpose => shape => [n_time*n_cond, row_count]
    D_avg_T = D_avg_2d.T

    # We can manually do PCA with SVD:
    # D_avg_T = U S Vt => principal components in Vt, scores in U, ...
    # But we also need the mean across the columns:
    mu = np.mean(D_avg_T, axis=0)  # shape [row_count]
    D_centered = D_avg_T - mu

    # SVD
    U_svd, S_svd, Vt_svd = np.linalg.svd(D_centered, full_matrices=False)
    # columns of Vt_svd^T = principal directions; i.e. V = Vt_svd.T
    # explained variance ratio:
    #   total_var = sum of squares of data_centered
    total_var = np.sum(D_centered**2)
    # variance per PC = (S_svd**2)
    var_pc = (S_svd**2) / (D_avg_T.shape[0] - 1)  # unbiased?
    # fraction explained by each PC
    explained = var_pc / (total_var / (D_avg_T.shape[0] - 1)) * 100.0
    varExp_pc = np.cumsum(explained)

    # 'coeff' in MATLAB is basically V: each column is a principal component
    coeff = Vt_svd.T  # shape [row_count, row_count]

    # ----------------------------------------------------
    # 3) Orthogonalize sub-blocks for each session 
    #    row_idxs_all define which rows belong to each session.
    # ----------------------------------------------------
    U_orth = orthogonalize_sub_matrices(coeff, row_idxs_all)

    # ----------------------------------------------------
    # 4) Align each session's data
    # ----------------------------------------------------
    nPCs_align = pars["alignment"]["nPCs_align"]
    for i_session in range(num_sessions):
        # Copy over the data
        D_align[i_session] = dict(data[i_session])  # shallow copy
        n_units, num_times, num_trials = data[i_session]["response"].shape
        
        # Subtract mu for the relevant rows
        session_rows = row_idxs_all[i_session]
        # mu[session_rows] has shape [n_units], but in Python it's a single dim. 
        # We want to subtract from each column in response(:,:). 
        # Flatten to [n_units, num_times*n_trials]
        resp_2d = D_align[i_session]["response"].reshape(n_units, -1)
        mean_sub_response_full = resp_2d - mu[session_rows][:, np.newaxis]

        # Align using first nPCs_align
        U_session = U_orth[i_session][:, :nPCs_align]  # shape [n_units, nPCs_align]
        aligned_response_2d = U_session.T @ mean_sub_response_full  # shape [nPCs_align, num_times*n_trials]

        # Reshape back
        aligned_response = aligned_response_2d.reshape(nPCs_align, num_times, num_trials)
        D_align[i_session]["response"] = aligned_response
        D_align[i_session]["U"] = U_session
        D_align[i_session]["X_mu"] = mu[session_rows]

        # For each dimension from 1..size(U_orth[i_session],2), compute varExp
        # replicating: varExp_ses{i_session}(idim) = compute_session_variance(..., U_orth{i_session}(:,1:idim))
        # We'll store an array of length = total dims in that sub-block:
        n_dims_session = U_orth[i_session].shape[1]
        varExp_ses[i_session] = np.zeros(n_dims_session, dtype=np.float64)
        for idim in range(n_dims_session):
            # U_orth[i_session][:, :idim+1] => shape [n_units, idim+1]
            chunk = U_orth[i_session][:, :idim+1]
            # Data: (D_avg[row_idxs_all{i_session}, :, :] - mu[session_rows]) 
            # flatten time/cond => shape [n_units, n_time*n_cond]
            session_block = D_avg[row_idxs_all[i_session], :, :]
            session_block_2d = session_block.reshape(n_units, -1)
            session_block_centered = session_block_2d - mu[session_rows][:, np.newaxis]

            varExp_ses[i_session][idim] = compute_session_variance(session_block_centered, chunk)

    # ----------------------------------------------------
    # 5) For plotting: D_cond_avg_proj_align
    #    We call sort_trials_by_condition on D_align[i_session] 
    #    with 'pars["align_proj"]["condition_vars"]'
    #    Then compute trial means. 
    #    Then we stack across sessions.
    # ----------------------------------------------------
    D_cond_avg_proj_list = []
    #print(D_align[0].keys())
    for i_session in range(num_sessions):
        #D_cond_align, task_conds_proj = sort_trials_by_condition(
        D_cond_align = sort_trials_by_condition(
            D_align[i_session],
            pars["align_proj"]["condition_vars"]
        )
        # cellfun(@(x) mean(x.response,3), D_cond_align, 'uni', false)
        # means => shape [nPCs_align, n_time]
        means_for_session = []
        for cdict in D_cond_align:
            means_for_session.append(np.mean(cdict["response"], axis=2))  # shape => [nPCs_align, n_time]
        # Stack along a new axis => shape [nPCs_align, n_time, n_cond_for_this_session]
        means_for_session = np.stack(means_for_session, axis=2)
        D_cond_avg_proj_list.append(means_for_session)

    # Now we cat these across sessions. The code does:
    # D_cond_avg_proj_align = cat(ndims(D_avg_align)+1, D_cond_avg_proj_align{:})
    # If the shape is [nPCs_align, n_time, n_cond], then ndims(D_avg_align) = 3, so we cat along dim=4. 
    # In Python, let's just stack along a new dimension => shape [nPCs_align, n_time, n_cond, n_sessions].
    D_cond_avg_proj_align = np.stack(D_cond_avg_proj_list, axis=3)

    # ----------------------------------------------------
    # 6) Compute alignment similarity
    # ----------------------------------------------------
    align_sim_med = compute_alignment_similarity(D_cond_avg_proj_align)

    # ----------------------------------------------------
    # 7) Build the stats output
    # ----------------------------------------------------
    align_stats = {
        "varExp_pc": varExp_pc,       # cumsum of explained variance across global PCA
        "varExp_ses": varExp_ses,     # per-session variance explained arrays
        "align_sim": align_sim_med,   # alignment similarity
    }

    return D_align, D_cond_avg_proj_align, align_stats #D_align, D_cond_avg_proj_align, task_conds_proj, align_stats


# ----------------------------------------------------------------
# Placeholders for functions used in the code above
# ----------------------------------------------------------------

def sort_trials_by_condition(data_dict, condition_vars):
    """
    Sort trials in data_dict by the unique combinations of
    the variables in `condition_vars`. Returns a list of data
    subsets (D_cond), plus some object (e.g. a DataFrame)
    describing the condition groups.

    This replicates your MATLAB function's behavior.
    """
    
    sortedT = []
    condition = data_dict['condition']
    uC = np.sort(np.unique(condition))
    for uCi in uC:
        w = np.where(condition==uCi)[0]
        d = deepcopy(data_dict)
        d['response']  = d['response'][:,:,w]
        d['condition'] = d['condition'][w]
        sortedT.append(d)
    
    return np.array(sortedT)


def myqr(A):
    """
    Placeholder for your custom 'myqr' function used in
    `orthogonalize_sub_matrices`.  For instance, this might
    just be np.linalg.qr(A).
    """
    # Typically:
    Q, R = np.linalg.qr(A)
    return Q, R
    #raise NotImplementedError


def orthogonalize_sub_matrices(U, unit_indices):
    """
    For each session, extract the sub-matrix U_seg = U[unit_indices[i_seg], :]
    and orthonormalize it. The result is U_orth[i_seg] = Q.

    This corresponds to your MATLAB subfunction:
      function [U_orth] = orthogonalize_sub_matrices(U,unit_indices)
    """
    num_sessions = len(unit_indices)
    U_orth = [None] * num_sessions
    for i_seg in range(num_sessions):
        rows = unit_indices[i_seg]
        U_seg = U[rows, :]  # shape [n_units_for_segment, rank]
        Q, _ = myqr(U_seg)
        U_orth[i_seg] = Q
    return U_orth


def compute_session_variance(D, U):
    """
    Replicates your MATLAB subfunction:
      var_exp = percvar(D, U*U'*D)
    i.e. the percentage of variance of D that is captured by U*U' * D.
    """
    # We can compute as percvar(D, approx) = 100 * (norm(approx, 'fro')^2 / norm(D, 'fro')^2)
    approx = U @ (U.T @ D)  # shape [n_units, n_time*n_cond]
    numerator = np.sum(approx**2)
    denominator = np.sum(D**2)
    var_exp = 100.0 * numerator / denominator
    return var_exp


def compute_alignment_similarity(D):
    """
    Replicates your MATLAB function compute_alignment_similarity(D).
    The shape of D is [n_modes, n_time, n_conditions, n_sessions].
    The code does:

        for i1 in 1..n_modes:
            for i2 in 1..n_modes:
                X = D(i1, :, :, :)
                Y = D(i2, :, :, :)
                M = corr(X(:,:)', Y(:,:)')
                M_med(i1,i2) = median(M(mask))

    We'll adapt that logic carefully.
    """
    # We assume D has shape (n_modes, n_time, n_conditions, n_sessions).
    n_modes = D.shape[0]
    M_med = np.full((n_modes, n_modes), np.nan)

    for i1 in range(n_modes):
        for i2 in range(n_modes):
            # X shape => [n_sessions, n_time, n_conditions]
            # But in MATLAB code, it does a permute to put sessions first, then time, then conditions
            # Let's replicate that carefully:
            X = np.transpose(D[i1, :, :, :], (2, 0, 1))  # shape => [n_sessions, n_time, n_conditions]
            Y = np.transpose(D[i2, :, :, :], (2, 0, 1))  # same shape

            # Now flatten each session's data for correlation. 
            # X(:,:)' => that means each row is a "flattened time+cond" dimension for that session.
            # So let's do X_sess => shape [n_sessions, (n_time*n_conditions)]
            # Then we want M = corr(X_sess', Y_sess')
            # so we do correlation across sessions dimension?
            # Actually from the code: M = corr(X(:,:)', Y(:,:)'), they flatten the second dimension across 
            #   n_time, n_conditions, leaving sessions as the first dimension. Then they correlate across sessions.
            # But the code does: 
            #     X = permute(squeeze(D(i1,:,:,:)), [3, 1, 2]) 
            # That yields shape => [n_sessions, n_time, n_conditions].
            # Then X(:,:) => shape => [n_sessions, n_time*n_conditions], 
            # so X(:,:)' => shape => [n_time*n_conditions, n_sessions].
            # So M = corr(...) => shape => [n_sessions, n_sessions].
            # Then they take the median of the upper-triangular part. 
            # We'll replicate exactly that:
            X_flat = X.reshape(X.shape[0], -1)  # shape => [n_sessions, n_time*n_conditions]
            Y_flat = Y.reshape(Y.shape[0], -1)

            # Compute correlation across sessions => M is [n_sessions, n_sessions]
            # We'll do a standard correlation across the "variables" dimension => axis=1. 
            # So let's transpose them so each row is a session. Then each column is the dimension we correlate across.
            # Actually it's simpler to compute the correlation matrix of X_flat, Y_flat rowwise:
            M = corr_matrix(X_flat, Y_flat)  # shape => [n_sessions, n_sessions]

            # We'll take the upper-triangular part (excluding diag) and compute median
            mask = np.triu_indices(M.shape[0], k=1)  # upper-triangular excluding diagonal
            # M_med(i1,i2) = median(M(mask));
            M_med[i1, i2] = np.median(M[mask])

    return M_med


def corr_matrix(A, B):
    """
    Compute the matrix of Pearson correlations between 
    every row of A and every row of B.
    A, B shape => [n_rows, n_features]
    Returns => corr of shape [n_rows, n_rows]
    """
    # Standard formula:
    # corr(A[i,:], B[j,:]) = cov(A[i,:], B[j,:]) / (std(A[i,:]) * std(B[j,:]))
    # We can do this in a vectorized way:
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)

    # Norm
    A_norm = np.sqrt(np.sum(A_centered**2, axis=1, keepdims=True))
    B_norm = np.sqrt(np.sum(B_centered**2, axis=1, keepdims=True))

    # Dot products => shape [nA, nB]
    dot_prod = A_centered @ B_centered.T
    corr_m = dot_prod / (A_norm @ B_norm.T + 1e-12)

    return corr_m

def test_existing_alignment(data, condition_vars):
    """
    Test alignment similarity without performing PCA or alignment,
    preserving unit structure (no projection into PCA space).
    
    Parameters
    ----------
    data : list of dict
        Each dict must contain:
            - 'response': np.ndarray [n_units, n_timepoints, n_trials]
            - 'condition': trial condition labels
    condition_vars : list
        Condition grouping variables (e.g., ['condition']).
    
    Returns
    -------
    align_sim : np.ndarray
        Alignment similarity matrix across units (n_units x n_units).
    """
    D_cond_avg_list = []

    for i_session in range(len(data)):
        # Sort trials by condition
        D_cond = sort_trials_by_condition(data[i_session], condition_vars)
        
        # Compute mean response for each condition: [n_units, n_time]
        means_for_session = [np.mean(cdict["response"], axis=2) for cdict in D_cond]
        
        # Stack into [n_units, n_time, n_conditions]
        means_for_session = np.stack(means_for_session, axis=2)
        D_cond_avg_list.append(means_for_session)

    # Stack across sessions => [n_units, n_time, n_conditions, n_sessions]
    D_cond_avg = np.stack(D_cond_avg_list, axis=3)

    # Run similarity computation
    align_sim = compute_alignment_similarity(D_cond_avg)

    return align_sim



def compute_session_alignment_CONTRAST_2BIAS_OLD(data, condition_vars, pars,t1,t2):
    """
    Align sessions using ONLY a single 'condition contrast' axis.

    Inputs
    ------
    data : list of dict
        Each session dict must contain:
          - "response": np.ndarray, shape (n_units, n_time, n_trials)

    condition_vars : list
        Passed to `sort_trials_by_condition` to split trials into conditions.

    pars : dict
        Used keys (with defaults):
          pars["alignment"]["cond_contrast"]   : (a_idx, b_idx), default (0,1) if n_cond>=2 else error
          pars["align_proj"]["condition_vars"] : list, optional (defaults to `condition_vars`)

    Behavior
    --------
    - No centering before projection (raw projections).
    - Uniform time weighting (mean over time).
    - Sign consistency: for each session, choose axis so that mean(A − B) > 0.
    - Bias term: choose scalar bias so that the (time- & trial-averaged) means of A/B
      are symmetric around 0, i.e., A > 0 and B < 0.

    Returns
    -------
    D_align : list of dict
        Mirrors `data`, but:
          - "response": (1, n_time, n_trials) aligned on the contrast axis and bias-shifted
          - "U":        (n_units, 1) per-session contrast axis (unit norm; sign-consistent)
          - "bias":     float, the scalar bias added to all projections in this session
          - "X_mu":     (n_units,) zeros (kept for API compatibility; not used)

    D_cond_avg_proj_align : np.ndarray
        Shape (1, n_time, n_cond, n_sessions): trial-averaged aligned trajectories.

    align_stats : dict
        {
          "cond_axis_present" : bool,
          "cond_contrast"     : (a_idx, b_idx),
          "time_weighting"    : "uniform"
        }
    """
    assert isinstance(data, list), "data must be a list of session dictionaries"
    assert isinstance(condition_vars, list), "condition_vars must be a list"

    alignment_pars  = pars.get("alignment", {})
    align_proj_vars = pars.get("align_proj", {}).get("condition_vars", condition_vars)

    num_sessions = len(data)
    row_idxs_all = [None] * num_sessions
    row_count = 0
    all_session_means = []

    # ---------- 1) Trial-averaged blocks per condition ----------
    for i_session in range(num_sessions):
        D_cond = sort_trials_by_condition(data[i_session], condition_vars)
        if len(D_cond) == 0:
            raise ValueError(f"Session {i_session} has no conditions after sorting.")
        n_units, n_time = D_cond[0]["response"].shape[:2]
        row_idxs_all[i_session] = np.arange(row_count, row_count + n_units)
        row_count += n_units

        means_this_session = np.stack(
            [np.mean(cdict["response"], axis=2) for cdict in D_cond],
            axis=2
        )  # (n_units, n_time, n_cond)
        all_session_means.append(means_this_session)

    n_conditions = all_session_means[0].shape[2]
    for i, ms in enumerate(all_session_means):
        if ms.shape[1:] != (n_time, n_conditions):
            raise ValueError(f"Session {i} has different n_time/n_cond; cannot stack.")

    # Choose contrast indices
    if "cond_contrast" in alignment_pars:
        a_idx, b_idx = alignment_pars["cond_contrast"]
    else:
        if n_conditions >= 2:
            a_idx, b_idx = (0, 1)
        else:
            raise ValueError("Need at least 2 conditions to form a contrast axis.")

    # ---------- 2) Align each session’s single-trial data ----------
    have_any_axis = False
    D_align = [None] * num_sessions

    for i_session, rows in enumerate(row_idxs_all):
        sess_means = all_session_means[i_session]                     # (n_units, n_time, n_cond)
        n_units, num_times, num_trials = data[i_session]["response"].shape

        # Uniform time weighting: mean over time of (A − B)
        local_diff = sess_means[:, t1:t2, a_idx] - sess_means[:, t1:t2, b_idx]  # (n_units, n_time)
        local_ref  = np.mean(local_diff, axis=1)                         # (n_units,)

        # Build axis; enforce sign so mean(A − B) > 0
        norm_ref = np.linalg.norm(local_ref)
        if norm_ref > 0:
            U_session = local_ref / norm_ref
            have_any_axis = True
            if float(np.dot(U_session, local_ref)) < 0:
                U_session = -U_session
        else:
            U_session = np.zeros(n_units, dtype=float)

        U_session = U_session[:, None]                                   # (n_units, 1)

        # Project raw single trials (NO centering)
        resp_2d    = data[i_session]["response"].reshape(n_units, -1)     # (n_units, T*trials)
        aligned_2d = U_session.T @ resp_2d                                # (1, T*trials)
        aligned    = aligned_2d.reshape(1, num_times, num_trials)         # (1, T, trials)

        # -------- Bias so that A > 0 and B < 0 on average --------
        # Compute time-averaged projected means of A and B using session means:
        proj_A_time = float(np.mean(U_session.T @ sess_means[:, t1:t2, a_idx]))  # scalar
        proj_B_time = float(np.mean(U_session.T @ sess_means[:, t1:t2, b_idx]))  # scalar

        if np.isfinite(proj_A_time) and np.isfinite(proj_B_time):
            # With sign consistency above, proj_A_time - proj_B_time >= 0
            bias = -0.5 * (proj_A_time + proj_B_time)
        else:
            bias = 0.0

        aligned = aligned + bias  # shift all trials equally

        # Package
        D_align[i_session] = dict(data[i_session])                        # shallow copy
        D_align[i_session]["response"] = aligned
        D_align[i_session]["U"] = U_session
        D_align[i_session]["bias"] = float(bias)
        D_align[i_session]["X_mu"] = np.zeros(n_units, dtype=float)       # placeholder for API compatibility

    # ---------- 3) Trial-averaged aligned data ----------
    D_cond_avg_proj_list = []
    for i_session in range(num_sessions):
        D_cond_align = sort_trials_by_condition(D_align[i_session], align_proj_vars)
        means_for_session = np.stack(
            [np.mean(cdict["response"], axis=2) for cdict in D_cond_align],
            axis=2
        )  # (1, n_time, n_cond)
        D_cond_avg_proj_list.append(means_for_session)

    D_cond_avg_proj_align = np.stack(D_cond_avg_proj_list, axis=3)        # (1, n_time, n_cond, n_sessions)

    # ---------- 4) Stats ----------
    align_stats = {
        "cond_axis_present": bool(have_any_axis),
        "cond_contrast": (a_idx, b_idx),
        "time_weighting": "uniform",
    }

    return D_align, D_cond_avg_proj_align, align_stats

import numpy as np

def compute_session_alignment_CONTRAST_2BIAS(data, condition_vars, pars, t1, t2):
    """
    Per-session alignment using a single condition-contrast axis.
    No global vector: each session gets its own U_session.
    """
    assert isinstance(data, list), "data must be a list of session dictionaries"
    assert isinstance(condition_vars, list), "condition_vars must be a list"

    alignment_pars  = pars.get("alignment", {})
    align_proj_vars = pars.get("align_proj", {}).get("condition_vars", condition_vars)

    num_sessions = len(data)
    D_align = [None] * num_sessions
    have_any_axis = False

    # ---------- 1) Compute per-session condition means ----------
    session_cond_means = []
    for i_session in range(num_sessions):
        D_cond = sort_trials_by_condition(data[i_session], condition_vars)
        if len(D_cond) == 0:
            raise ValueError(f"Session {i_session} has no conditions after sorting.")

        # (n_units, n_time, n_cond) where each slice is trial-mean for that condition
        means_this_session = np.stack(
            [np.mean(cdict["response"], axis=2) for cdict in D_cond],
            axis=2
        )
        session_cond_means.append(means_this_session)

    # Basic consistency checks + get n_conditions
    n_conditions = session_cond_means[0].shape[2]
    n_time0 = session_cond_means[0].shape[1]
    for i, ms in enumerate(session_cond_means):
        if ms.shape[1] != n_time0 or ms.shape[2] != n_conditions:
            raise ValueError(f"Session {i} has different n_time/n_cond; cannot proceed.")

    # Choose contrast indices
    if "cond_contrast" in alignment_pars:
        a_idx, b_idx = alignment_pars["cond_contrast"]
    else:
        if n_conditions >= 2:
            a_idx, b_idx = (0, 1)
        else:
            raise ValueError("Need at least 2 conditions to form a contrast axis.")

    # ---------- 2) Align each session with its own axis ----------
    for i_session in range(num_sessions):
        sess_means = session_cond_means[i_session]  # (n_units, n_time, n_cond)
        n_units, num_times, num_trials = data[i_session]["response"].shape

        # Uniform time weighting: mean over time of (A − B)
        local_diff = sess_means[:, t1:t2, a_idx] - sess_means[:, t1:t2, b_idx]  # (n_units, t_window)
        local_ref  = np.mean(local_diff, axis=1)                                # (n_units,)

        # Build per-session axis
        norm_ref = np.linalg.norm(local_ref)
        if norm_ref > 0:
            U_session = local_ref / norm_ref
            have_any_axis = True
            # sign consistency so mean(A − B) > 0
            if float(np.dot(U_session, local_ref)) < 0:
                U_session = -U_session
        else:
            U_session = np.zeros(n_units, dtype=float)

        U_session = U_session[:, None]  # (n_units, 1)

        # Project raw single trials (NO centering)
        resp_2d    = data[i_session]["response"].reshape(n_units, -1)  # (n_units, T*trials)
        aligned_2d = U_session.T @ resp_2d                             # (1, T*trials)
        aligned    = aligned_2d.reshape(1, num_times, num_trials)      # (1, T, trials)

        # -------- Bias so that A > 0 and B < 0 on average --------
        proj_A_time = float(np.mean(U_session.T @ sess_means[:, t1:t2, a_idx]))  # scalar
        proj_B_time = float(np.mean(U_session.T @ sess_means[:, t1:t2, b_idx]))  # scalar

        if np.isfinite(proj_A_time) and np.isfinite(proj_B_time):
            bias = -0.5 * (proj_A_time + proj_B_time)
        else:
            bias = 0.0

        aligned = aligned + bias

        # Package
        D_align[i_session] = dict(data[i_session])  # shallow copy
        D_align[i_session]["response"] = aligned
        D_align[i_session]["U"] = U_session
        D_align[i_session]["bias"] = float(bias)
        D_align[i_session]["X_mu"] = np.zeros(n_units, dtype=float)  # kept for API compatibility

    # ---------- 3) Trial-averaged aligned data ----------
    D_cond_avg_proj_list = []
    for i_session in range(num_sessions):
        D_cond_align = sort_trials_by_condition(D_align[i_session], align_proj_vars)
        means_for_session = np.stack(
            [np.mean(cdict["response"], axis=2) for cdict in D_cond_align],
            axis=2
        )  # (1, n_time, n_cond)
        D_cond_avg_proj_list.append(means_for_session)

    D_cond_avg_proj_align = np.stack(D_cond_avg_proj_list, axis=3)  # (1, n_time, n_cond, n_sessions)

    align_stats = {
        "cond_axis_present": bool(have_any_axis),
        "cond_contrast": (a_idx, b_idx),
        "time_weighting": "uniform",
    }

    return D_align, D_cond_avg_proj_align, align_stats
