import os
import time
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict, Counter
from pymatgen.core.structure import Structure
import pure_peak
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

def calinski_harabasz_from_distance(dist_matrix, labels):
    n = len(labels)
    k = len(np.unique(labels))
    if k <= 1 or k >= n:
        return 0
    total_ss = np.sum(dist_matrix[np.triu_indices(n, k=1)])
    wss = 0
    for cid in np.unique(labels):
        mask = labels == cid
        n_k = np.sum(mask)
        if n_k < 2:
            continue
        sub_matrix = dist_matrix[np.ix_(mask, mask)]
        triu_indices = np.triu_indices_from(sub_matrix, k=1)
        wss += np.sum(sub_matrix[triu_indices])
    bss = total_ss - wss
    ch_index = (bss / (k - 1)) / (wss / (n - k)) if wss > 0 else 0
    return ch_index

def davies_bouldin_from_distance(dist_matrix, labels):
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1:
        return 0
    S = {}
    for cid in unique:
        mask = labels == cid
        n_k = np.sum(mask)
        if n_k < 2:
            S[cid] = 0
            continue
        sub = dist_matrix[np.ix_(mask, mask)]
        triu = np.triu_indices_from(sub, k=1)
        S[cid] = np.mean(sub[triu])
    d = {}
    for i, ci in enumerate(unique):
        for cj in unique[i + 1:]:
            mask_i = labels == ci
            mask_j = labels == cj
            sub = dist_matrix[np.ix_(mask_i, mask_j)]
            meanij = np.mean(sub)
            d[(ci, cj)] = meanij
            d[(cj, ci)] = meanij
    R_max = []
    for ci in unique:
        Rij = []
        for cj in unique:
            if ci == cj:
                continue
            denom = d.get((ci, cj), 0)
            if denom > 0:
                Rij.append((S[ci] + S[cj]) / denom)
        if Rij:
            R_max.append(max(Rij))
    return np.mean(R_max) if R_max else 0

def sse_from_distance_to_medoid(dist_matrix, labels):
    sse = 0.0
    for cid in np.unique(labels):
        mask = labels == cid
        idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            continue
        sub = dist_matrix[np.ix_(idx_arr, idx_arr)]
        if sub.shape[0] == 1:
            continue
        dist_sums = sub.sum(axis=1)
        medoid_local = int(np.argmin(dist_sums))
        medoid_dists = sub[:, medoid_local]
        sse += float(np.sum(np.square(medoid_dists)))
    return sse
    
def separation_ratio_from_distance(dist_matrix, labels):
    """
    Calculate the separation ratio using cluster-level macro averaging:

    separation ratio = mean inter-cluster distance / mean intra-cluster distance

    For each cluster:
    - The intra-cluster distance is the mean pairwise distance within the cluster.
    - A singleton cluster is retained and its intra-cluster distance is explicitly set to 0.
    - The inter-cluster distance is the mean distance from all samples in the
      current cluster to all samples outside that cluster.

    The final intra-cluster and inter-cluster distances are arithmetic means
    over all clusters, including singleton clusters. This definition is kept
    consistent with 4108_clustering.py.
    """
    unique_labels = np.unique(labels)
    intra_dists = []
    inter_dists = []

    for lab in unique_labels:
        idx_in = np.where(labels == lab)[0]
        idx_out = np.where(labels != lab)[0]
        cluster_size = len(idx_in)

        # Retain singleton clusters and explicitly assign an intra-cluster
        # distance of 0. Multi-member clusters use their mean pairwise distance.
        if cluster_size < 2:
            intra_dists.append(0.0)
        else:
            sub_in = dist_matrix[np.ix_(idx_in, idx_in)]
            triu = np.triu_indices_from(sub_in, k=1)
            if len(triu[0]) > 0:
                intra_dists.append(float(np.mean(sub_in[triu])))
            else:
                intra_dists.append(0.0)

        # Mean distance from the current cluster to all samples outside it.
        if len(idx_out) > 0:
            sub_out = dist_matrix[np.ix_(idx_in, idx_out)]
            inter_dists.append(float(np.mean(sub_out)))
        else:
            # There is no inter-cluster distance when only one cluster exists.
            inter_dists.append(0.0)

    if not intra_dists or not inter_dists:
        return np.nan

    mean_intra = float(np.mean(intra_dists))
    mean_inter = float(np.mean(inter_dists))
    return mean_inter / mean_intra if mean_intra > 0 else np.nan

def save_cluster_quality_details(outdir, dist_matrix, labels):
    """
    Save per-cluster distance statistics.

    Singleton clusters are retained. Their intra-cluster mean and standard
    deviation are set to 0 because no distinct within-cluster pair exists.
    Their per-cluster separation ratio is also set to 0 to avoid division by
    zero in this descriptive table. This does not change the global
    Separation_Ratio reported in metrics.csv.
    """
    rows = []
    unique_labels = np.unique(labels)

    for cid in unique_labels:
        idx_in = np.where(labels == cid)[0]
        idx_out = np.where(labels != cid)[0]
        cluster_size = len(idx_in)

        if cluster_size >= 2:
            intra_matrix = dist_matrix[np.ix_(idx_in, idx_in)]
            intra_indices = np.triu_indices_from(intra_matrix, k=1)
            intra_values = intra_matrix[intra_indices]
            intra_mean = float(np.mean(intra_values))
            intra_std = float(np.std(intra_values))
        else:
            intra_mean = 0.0
            intra_std = 0.0

        if len(idx_out) > 0:
            inter_values = dist_matrix[np.ix_(idx_in, idx_out)].ravel()
            inter_mean = float(np.mean(inter_values))
            inter_std = float(np.std(inter_values))
        else:
            inter_mean = 0.0
            inter_std = 0.0

        cluster_separation_ratio = inter_mean / intra_mean if intra_mean > 0 else 0.0

        rows.append({
            'cluster_id': int(cid),
            'Size': int(cluster_size),
            'Intra_Distance_Mean': intra_mean,
            'Intra_Distance_Std': intra_std,
            'Inter_Distance_Mean': inter_mean,
            'Inter_Distance_Std': inter_std,
            'Separation_Ratio': cluster_separation_ratio
        })

    output_path = os.path.join(outdir, 'cluster_quality_details.csv')
    pd.DataFrame(rows, columns=[
        'cluster_id',
        'Size',
        'Intra_Distance_Mean',
        'Intra_Distance_Std',
        'Inter_Distance_Mean',
        'Inter_Distance_Std',
        'Separation_Ratio'
    ]).to_csv(output_path, index=False)


def crystal_system_from_space_group(space_group_number):
    """Return the crystal system for an international space-group number."""
    number = int(space_group_number)
    if 1 <= number <= 2:
        return 'triclinic'
    if 3 <= number <= 15:
        return 'monoclinic'
    if 16 <= number <= 74:
        return 'orthorhombic'
    if 75 <= number <= 142:
        return 'tetragonal'
    if 143 <= number <= 167:
        return 'trigonal'
    if 168 <= number <= 194:
        return 'hexagonal'
    if 195 <= number <= 230:
        return 'cubic'
    raise ValueError(
        f'Invalid international space-group number: {space_group_number}'
    )


def build_compound_metadata(names, cif_dir):
    """
    Build display names, space-group numbers, and crystal systems.

    If a frozen matrix name already ends in ``_<space-group number>``, that
    suffix is used directly. Otherwise, the corresponding CIF file is read,
    its space-group number is determined, and the number is appended to the
    display name.

    The iteration follows the frozen ``names`` sequence exactly; glob() is not
    used to reconstruct or reorder matrix identities.
    """
    metadata = {}
    suffix_pattern = re.compile(r'_(\d{1,3})$')

    for name in names:
        match = suffix_pattern.search(name)
        if match is not None:
            space_group_number = int(match.group(1))
            compound_display_name = name
        else:
            cif_path = os.path.join(cif_dir, f'{name}.cif')
            if not os.path.exists(cif_path):
                raise FileNotFoundError(
                    'Cannot determine the space group because the CIF file '
                    f'was not found: {os.path.abspath(cif_path)}'
                )
            structure = Structure.from_file(cif_path)
            _, space_group_number = structure.get_space_group_info()
            space_group_number = int(space_group_number)
            compound_display_name = f'{name}_{space_group_number}'

        # This validates that the suffix or CIF-derived number is in 1..230.
        crystal_system = crystal_system_from_space_group(space_group_number)
        metadata[name] = {
            'compound_name': compound_display_name,
            'space_group_number': space_group_number,
            'crystal_system': crystal_system
        }

    return metadata


def save_compound_cluster_metadata(
    outdir,
    labels,
    names,
    cluster_to_medoid,
    compound_metadata
):
    """
    Save compound metadata using the unified cluster IDs.

    ``labels`` has already been relabeled according to the Python ``sorted()``
    order of the original medoid names. Therefore, this function must not sort
    or relabel clusters again. The ``cluster_id`` written here is identical to
    the IDs in clusters.csv, prototypes.csv, and cluster_quality_details.csv.
    """
    rows = []
    for matrix_index, name in enumerate(names):
        cluster_id = int(labels[matrix_index])
        representative_name = cluster_to_medoid[cluster_id]
        member_metadata = compound_metadata[name]
        representative_metadata = compound_metadata[representative_name]

        rows.append({
            'Compound': member_metadata['compound_name'],
            'Space_Group': member_metadata['space_group_number'],
            'Crystal_System': member_metadata['crystal_system'],
            'cluster_id': cluster_id,
            'Representative_Compound': representative_metadata['compound_name']
        })

    output_columns = [
        'Compound',
        'Space_Group',
        'Crystal_System',
        'cluster_id',
        'Representative_Compound'
    ]
    output_path = os.path.join(outdir, 'compound_cluster_summary.csv')
    output_df = pd.DataFrame(rows, columns=output_columns).sort_values(
        by='Compound',
        key=lambda series: series.astype(str).str.casefold(),
        kind='stable'
    )
    # utf-8-sig allows the English-header CSV to open cleanly in Excel.
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def compute_cluster_medoids(labels, names, dist_mat):
    """Return ``cluster_id -> medoid name`` for the supplied labels."""
    cluster_to_indices = defaultdict(list)
    for matrix_index, cluster_id in enumerate(labels):
        cluster_to_indices[int(cluster_id)].append(matrix_index)

    cluster_to_medoid = {}
    for cluster_id, member_indices in cluster_to_indices.items():
        idx_arr = np.asarray(member_indices, dtype=int)
        sub_dist = dist_mat[np.ix_(idx_arr, idx_arr)]
        distance_sums = sub_dist.sum(axis=1)
        medoid_local_index = int(np.argmin(distance_sums))
        medoid_global_index = int(idx_arr[medoid_local_index])
        cluster_to_medoid[int(cluster_id)] = names[medoid_global_index]

    return cluster_to_medoid


def relabel_clusters_by_medoid_name(labels, names, dist_mat):
    """
    Create the single cluster-ID system used by every output file.

    The original sklearn clusters are ordered by their medoid names using the
    same case-sensitive Python string ordering as ``sorted(representatives)``.
    The first medoid receives cluster_id 0, the second receives 1, and so on.

    Returns
    -------
    relabeled_labels : np.ndarray
        Per-sample labels using the unified contiguous cluster IDs.
    original_to_unified : dict
        Mapping from sklearn's original cluster ID to the unified cluster ID.
    original_cluster_to_medoid : dict
        Medoid names indexed by sklearn's original cluster IDs.
    """
    original_labels = np.asarray(labels, dtype=int)
    original_cluster_to_medoid = compute_cluster_medoids(
        original_labels, names, dist_mat
    )

    sorted_original_cluster_ids = sorted(
        original_cluster_to_medoid,
        key=lambda cluster_id: original_cluster_to_medoid[cluster_id]
    )
    original_to_unified = {
        int(original_cluster_id): int(unified_cluster_id)
        for unified_cluster_id, original_cluster_id
        in enumerate(sorted_original_cluster_ids)
    }

    relabeled_labels = np.asarray(
        [original_to_unified[int(cluster_id)] for cluster_id in original_labels],
        dtype=int
    )
    return (
        relabeled_labels,
        original_to_unified,
        original_cluster_to_medoid
    )


def save_cluster_id_mapping(
    outdir,
    original_to_unified,
    original_cluster_to_medoid,
    compound_metadata
):
    """Save an audit table connecting sklearn IDs to unified cluster IDs."""
    rows = []
    for original_cluster_id, cluster_id in sorted(
        original_to_unified.items(),
        key=lambda item: item[1]
    ):
        medoid_name = original_cluster_to_medoid[original_cluster_id]
        rows.append({
            'sklearn_cluster_id': int(original_cluster_id),
            'cluster_id': int(cluster_id),
            'medoid_name': medoid_name,
            'Representative_Compound': compound_metadata[medoid_name]['compound_name']
        })

    pd.DataFrame(rows, columns=[
        'sklearn_cluster_id',
        'cluster_id',
        'medoid_name',
        'Representative_Compound'
    ]).to_csv(
        os.path.join(outdir, 'cluster_id_mapping.csv'),
        index=False,
        encoding='utf-8-sig'
    )

def build_xrd_vectors(cif_dir, num_spectra=1, domain_size=25, theta_min=10, theta_max=80):
    theta_grid = np.linspace(theta_min, theta_max, 4501)

    # Sort CIF paths so that newly generated distance matrices use a
    # deterministic row/column order. The exact order is also saved separately
    # and must be reused whenever the matrix is loaded.
    cif_paths = sorted(glob(os.path.join(cif_dir, '*.cif')))
    if not cif_paths:
        raise FileNotFoundError(f'No CIF files were found in: {os.path.abspath(cif_dir)}')

    names = [os.path.basename(path).replace('.cif', '') for path in cif_paths]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate CIF base names were found; matrix indices would be ambiguous.')

    xrd_vectors = {}
    for name, cif_path in zip(names, cif_paths):
        struct = Structure.from_file(cif_path)
        pattern = pure_peak.main(struct, num_spectra, domain_size, theta_min, theta_max)
        xrd_vectors[name] = np.squeeze(pattern)

    X = np.vstack([xrd_vectors[name] for name in names])
    X_clipped = np.clip(X, 0, None)
    row_sums = X_clipped.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.squeeze() == 0)
    if np.any(zero_rows):
        n_theta = X_clipped.shape[1]
        X_clipped[zero_rows, :] = 1.0 / n_theta
        row_sums = X_clipped.sum(axis=1, keepdims=True)
    X_norm = X_clipped / row_sums
    return names, X_norm, theta_grid


def save_distance_matrix_index(names, names_path):
    """Save the exact name order corresponding to distance-matrix indices."""
    index_df = pd.DataFrame({
        'matrix_index': np.arange(len(names), dtype=int),
        'name': names
    })
    index_df.to_csv(names_path, index=False)


def load_distance_matrix_index(names_path, matrix_size):
    """Load and validate the frozen name order for a distance matrix."""
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            'The distance matrix exists, but its frozen name-index file is missing:\n'
            f'  {os.path.abspath(names_path)}\n'
            'The original row/column-to-compound mapping cannot be recovered safely '
            'from the current CIF directory order. Re-run this script once with '
            '--recompute-dist to regenerate the matrix and its matching index file.'
        )

    names_df = pd.read_csv(names_path)
    required_columns = {'matrix_index', 'name'}
    missing_columns = required_columns.difference(names_df.columns)
    if missing_columns:
        raise ValueError(
            f'Name-index file is missing required columns: {sorted(missing_columns)}'
        )

    if len(names_df) != matrix_size:
        raise ValueError(
            f'Name-index row count ({len(names_df)}) does not match '
            f'distance matrix size ({matrix_size}).'
        )

    expected_indices = np.arange(matrix_size, dtype=int)
    actual_indices = names_df['matrix_index'].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            'The matrix_index column must contain consecutive indices from 0 '
            f'to {matrix_size - 1} in exactly that order.'
        )

    names = names_df['name'].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate names were found in the name-index file.')

    return names

def wdist_factory(theta_grid):
    def wdist(u, v):
        return wasserstein_distance(theta_grid, theta_grid, u, v)
    return wdist

def save_clusters_and_prototypes(outdir, labels, names, dist_mat):
    """Save unified cluster assignments and medoids, then return the medoid map."""
    cluster_to_indices = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_to_indices[int(lab)].append(idx)

    prototype_rows = []
    cluster_rows = []
    cluster_to_medoid = {}

    for lab, idx_list in sorted(cluster_to_indices.items(), key=lambda item: item[0]):
        idx_arr = np.asarray(idx_list, dtype=int)
        member_names = [names[i] for i in idx_arr]
        sub = dist_mat[np.ix_(idx_arr, idx_arr)]

        # Every fitted cluster contains at least one sample. The singleton case
        # naturally produces [[0]], so its only member is selected as medoid.
        distance_sums = sub.sum(axis=1)
        medoid_local = int(np.argmin(distance_sums))
        medoid_global = int(idx_arr[medoid_local])
        medoid_name = names[medoid_global]

        cluster_to_medoid[lab] = medoid_name
        prototype_rows.append((lab, medoid_global, medoid_name))

        for member_index, member_name in zip(idx_arr, member_names):
            cluster_rows.append((lab, int(member_index), member_name))

    pd.DataFrame(
        prototype_rows,
        columns=['cluster_id', 'medoid_index', 'medoid_name']
    ).to_csv(os.path.join(outdir, 'prototypes.csv'), index=False)

    pd.DataFrame(
        cluster_rows,
        columns=['cluster_id', 'member_index', 'member_name']
    ).to_csv(os.path.join(outdir, 'clusters.csv'), index=False)

    return cluster_to_medoid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='results_by_tau', help='Root output directory')
    parser.add_argument('--recompute-dist', action='store_true', help='Recompute the distance matrix')
    parser.add_argument('--n_jobs', type=int, default=-1, help='pairwise_distances n_jobs')
    parser.add_argument('--cif_dir', default='../mp_perov_cif6')
    args = parser.parse_args()
    percentiles = range(5, 81, 5)

    os.makedirs(args.outdir, exist_ok=True)
    dist_path = 'perov_wasserstein_4108x4108.csv'
    names_path = os.path.splitext(dist_path)[0] + '_names.csv'

    print(f'Distance matrix path: {os.path.abspath(dist_path)}')
    print(f'Matrix name-index path: {os.path.abspath(names_path)}')

    if args.recompute_dist or not os.path.exists(dist_path):
        print('Calculating XRD vectors and generating the distance matrix...')
        names, X_norm, theta_grid = build_xrd_vectors(args.cif_dir)
        wdist = wdist_factory(theta_grid)
        t0 = time.time()
        dist_mat = pairwise_distances(X_norm, metric=wdist, n_jobs=args.n_jobs)

        # Save the matrix and the exact row/column-to-name mapping together.
        np.savetxt(dist_path, dist_mat, delimiter=',')
        save_distance_matrix_index(names, names_path)

        print(
            f'Distance matrix saved to {os.path.abspath(dist_path)}; '
            f'name index saved to {os.path.abspath(names_path)}; '
            f'elapsed time: {time.time()-t0:.1f}s'
        )
    else:
        print('Loading the distance matrix and its frozen name index...')
        dist_mat = np.loadtxt(dist_path, delimiter=',')

        if dist_mat.ndim != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
            raise ValueError(
                f'Distance matrix must be square, but its shape is {dist_mat.shape}.'
            )

        # Never rebuild names from glob() here. The saved index is the only
        # trustworthy mapping between matrix rows/columns and compounds.
        names = load_distance_matrix_index(names_path, dist_mat.shape[0])

    n = dist_mat.shape[0]
    if len(names) != n:
        raise ValueError(
            f'Name count ({len(names)}) does not match distance matrix size ({n}).'
        )
    if not np.all(np.isfinite(dist_mat)):
        print('Warning: the distance matrix contains NaN/Inf values; replacing them with finite values.')
        finite_mask = np.isfinite(dist_mat)
        if np.any(finite_mask):
            max_finite = np.nanmax(dist_mat[finite_mask])
            fill_val = max_finite * 10 if np.isfinite(max_finite) and max_finite > 0 else 1e6
        else:
            fill_val = 1e6
        dist_mat[~finite_mask] = fill_val

    np.fill_diagonal(dist_mat, 0.0)
    if not np.allclose(dist_mat, dist_mat.T, atol=1e-8):
        dist_mat = (dist_mat + dist_mat.T) / 2.0

    print('Building compound, space-group, and crystal-system metadata...')
    compound_metadata = build_compound_metadata(names, args.cif_dir)

    dists = dist_mat[np.triu_indices_from(dist_mat, k=1)]
    all_metrics = []

    for p in percentiles:
        tau = float(np.percentile(dists, p))
        subdir = os.path.join(args.outdir, f'p{int(p):02d}_tau_{tau:.4f}')
        os.makedirs(subdir, exist_ok=True)
        print(f'[{p}%] tau={tau:.4f} -> saving to {subdir}')

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=tau,
            compute_distances=True
        )
        sklearn_labels = clusterer.fit_predict(dist_mat)

        # Build one unified label system before calculating metrics or saving
        # files. IDs are assigned by sorted original medoid names, matching the
        # historical ``sorted(representatives)`` labeling rule.
        (
            labels,
            original_to_unified,
            original_cluster_to_medoid
        ) = relabel_clusters_by_medoid_name(
            sklearn_labels,
            names,
            dist_mat
        )

        unique = np.unique(labels)
        n_clusters = len(unique)
        expected_ids = np.arange(n_clusters, dtype=int)
        if not np.array_equal(unique, expected_ids):
            raise RuntimeError(
                'Unified cluster IDs are not contiguous from 0 to '
                f'{n_clusters - 1}: {unique.tolist()}'
            )

        sizes = Counter(labels)
        min_size = int(min(sizes.values()))
        max_size = int(max(sizes.values()))

        ch = calinski_harabasz_from_distance(dist_mat, labels)
        dbi = davies_bouldin_from_distance(dist_mat, labels)
        sse = sse_from_distance_to_medoid(dist_mat, labels)
        sep = separation_ratio_from_distance(dist_mat, labels)

        cluster_to_medoid = save_clusters_and_prototypes(
            subdir,
            labels,
            names,
            dist_mat
        )
        save_cluster_id_mapping(
            subdir,
            original_to_unified,
            original_cluster_to_medoid,
            compound_metadata
        )
        save_cluster_quality_details(subdir, dist_mat, labels)
        save_compound_cluster_metadata(
            subdir,
            labels,
            names,
            cluster_to_medoid,
            compound_metadata
        )

        # Internal consistency check: every saved cluster map must use exactly
        # the same unified IDs as the per-sample labels.
        if set(cluster_to_medoid) != set(unique.tolist()):
            raise RuntimeError(
                'Cluster-ID inconsistency detected between labels and medoids.'
            )

        metrics = {
            'percentile': p,
            'tau': tau,
            'n_samples': n,
            'n_clusters': n_clusters,
            'min_cluster_size': min_size,
            'max_cluster_size': max_size,
            'CH_index': ch,
            'DB_index': dbi,
            'SSE': sse,
            'Separation_Ratio': sep
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(subdir, 'metrics.csv'), index=False)
        all_metrics.append(metrics)
        print(f"  clusters={n_clusters}, min_size={min_size}, sse={sse:.4f}")

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = os.path.join(args.outdir, 'all_tau_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f'Summary saved to: {summary_csv}')

    # ================= Plotting section =================
    # 1. Set global font sizes
    plt.rcParams['axes.labelsize'] = 10   # Axis-label font size (10 pt)
    plt.rcParams['xtick.labelsize'] = 7  # X-axis tick-label font size (7 pt)
    plt.rcParams['ytick.labelsize'] = 7  # Y-axis tick-label font size (7 pt)
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['legend.fontsize'] = 10  # Legend font size (10 pt)

    # 2. Convert the 8 cm x 6 cm figure size to inches (1 inch = 2.54 cm)
    figsize_inches = (8 / 2.54, 5 / 2.54)

    # 3. Plot each evaluation metric against the number of clusters
    # Note: The original plot of n_clusters versus percentile was removed because using the number of clusters on both axes would produce a meaningless straight line.
    for metric_name in ['CH_index', 'DB_index', 'SSE', 'Separation_Ratio']:
        plt.figure(figsize=figsize_inches)
        # Use n_clusters as the x-axis
        plt.plot(summary_df['n_clusters'], summary_df[metric_name], marker='o', markersize=3, linewidth=1.0)
        plt.xlabel('Number of clusters')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs Number of clusters')
        plt.grid(True, alpha=0.3)
        # Update the output filename to reflect the x-axis definition
        plt.savefig(os.path.join(args.outdir, f'{metric_name}_vs_n_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # ================================================

    print('All tasks completed.')

if __name__ == '__main__':
    main()