import os.path
import sys
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
from collections import defaultdict
from Source.explainers.subgraphX.subgraphx import SubgraphX

sys.path.append(os.path.abspath("../../../"))
RDLogger.DisableLog('rdApp.*')


def get_subgraphs(mol, featurizer, explainable_model, device, subgraphX_kwargs, target_ids=(0,)):
    subgraphX_kwargs["device"] = device
    graph = featurizer.featurize(mol)

    explainer = SubgraphX(
        explainable_model,
        target_ids=target_ids,
        **subgraphX_kwargs
    )

    graph.to(device)
    _, explanation_results, _ = explainer(graph.x,
                                          graph.edge_index,
                                          max_nodes=graph.x.shape[0])

    subgraphs = explainer.read_from_MCTSInfo_list(explanation_results[0])

    return subgraphs


def get_subgraphX_scores(mol, featurizer, explainable_model, device, subgraphX_kwargs, target_ids=(0,)):
    subgraphX_kwargs["device"] = device
    graph = featurizer.featurize(mol)

    explainer = SubgraphX(
        explainable_model,
        target_ids=target_ids,
        **subgraphX_kwargs
    )

    graph.to(device)
    _, explanation_results, related_preds = explainer(graph.x,
                                          graph.edge_index,
                                          max_nodes=graph.x.shape[0])


    return explanation_results


def draw_subgraphs(mol, subgraphs):
    img = Chem.Draw.MolsToGridImage(
        [mol for _ in subgraphs],
        highlightAtomLists=[subgraph.coalition for subgraph in subgraphs],
        legends=[str(subgraph.P) for subgraph in subgraphs],
    )
    return img


def draw_best_subgraph(mol, subgraphs, max_nodes, show_value=True):
    subgraphs = [s for s in subgraphs if len(s.coalition) <= max_nodes]
    best_subgraph = subgraphs[0]
    legend = str(best_subgraph.P) if show_value else None
    img = Chem.Draw.MolToImage(mol, highlightAtoms=best_subgraph.coalition, legend=legend)
    return img


def calculate_atom_scores_multiple_coalitions(subgraphx_results_list, num_atoms, method='union'):
    """
    Calculate atom scores from multiple SubgraphX coalitions for a single molecule.
    
    Parameters:
    - subgraphx_results_list: List of dictionaries containing SubgraphX results for one molecule
    - num_atoms: Number of atoms in the molecule
    - method: How to combine multiple coalitions:
        'union' - atom is important if it appears in ANY coalition (default)
        'intersection' - atom is important only if it appears in ALL coalitions
        'frequency' - atom score equals frequency of appearance across coalitions (0-1)
        'binary_majority' - atom is important if it appears in >50% of coalitions
        'weighted' - weighted by SubgraphX P score if available
    
    Returns:
    - atom_scores: List of scores for each atom (binary or continuous)
    - important_atoms: List of atom indices that are important
    - stats: Dictionary with comprehensive statistics
    """
    
    if not isinstance(subgraphx_results_list, list):
        subgraphx_results_list = [subgraphx_results_list]
    
    # Initialize tracking arrays
    atom_frequency = [0] * num_atoms
    atom_weighted_scores = [0.0] * num_atoms
    all_coalitions = []
    p_scores = []
    
    # Process each coalition
    for i, result in enumerate(subgraphx_results_list):
        coalition = extract_coalition_from_result(result)
        
        if coalition is None:
            print(f"Warning: Could not find coalition in result {i}")
            continue
            
        all_coalitions.append(coalition)
        
        # Get P score if available (default to 1.0 if not found)
        p_score = result.get('P', 1.0)
        p_scores.append(p_score)
        
        # Update frequency and weighted scores
        for atom_idx in coalition:
            if atom_idx < num_atoms:
                atom_frequency[atom_idx] += 1
                atom_weighted_scores[atom_idx] += p_score
            else:
                print(f"Warning: Atom index {atom_idx} out of bounds for molecule with {num_atoms} atoms")
    
    # Calculate atom scores based on the selected method
    atom_scores = combine_coalitions(atom_frequency, atom_weighted_scores, 
                                   len(all_coalitions), method, p_scores)
    
    # Get important atoms (non-zero scores)
    if method in ['frequency', 'weighted']:
        # For continuous scores, use a threshold or return all with scores > 0
        important_atoms = [idx for idx, score in enumerate(atom_scores) if score > 0]
    else:
        # For binary methods
        important_atoms = [idx for idx, score in enumerate(atom_scores) if score == 1]
    
    # Calculate comprehensive statistics
    stats = calculate_comprehensive_stats(atom_scores, atom_frequency, all_coalitions, 
                                        num_atoms, method, p_scores)
    
    return atom_scores, important_atoms, stats

def extract_coalition_from_result(subgraphx_result):
    """
    Extract coalition from SubgraphX result dictionary.
    """
    # Try different possible keys for coalition
    if 'coalition' in subgraphx_result:
        return subgraphx_result['coalition']
    elif 'nodes' in subgraphx_result:
        return subgraphx_result['nodes']
    elif 'subgraph' in subgraphx_result:
        return subgraphx_result['subgraph']
    elif 'explanation_nodes' in subgraphx_result:
        return subgraphx_result['explanation_nodes']
    
    # If no direct coalition key, look for it in nested structure
    for key in subgraphx_result.keys():
        value = subgraphx_result[key]
        if hasattr(value, 'coalition'):
            return value.coalition
    
    # Last resort: try to find any list of nodes in the dictionary
    for key, value in subgraphx_result.items():
        if isinstance(value, (list, set)) and all(isinstance(x, int) for x in value):
            return value
    
    return None

def combine_coalitions(atom_frequency, atom_weighted_scores, num_coalitions, method, p_scores):
    """
    Combine multiple coalitions using the specified method.
    """
    if num_coalitions == 0:
        return [0] * len(atom_frequency)
    
    if method == 'union':
        # Atom is important if it appears in ANY coalition
        return [1 if freq > 0 else 0 for freq in atom_frequency]
    
    elif method == 'intersection':
        # Atom is important only if it appears in ALL coalitions
        return [1 if freq == num_coalitions else 0 for freq in atom_frequency]
    
    elif method == 'binary_majority':
        # Atom is important if it appears in >50% of coalitions
        threshold = num_coalitions / 2
        return [1 if freq > threshold else 0 for freq in atom_frequency]
    
    elif method == 'frequency':
        # Atom score equals frequency of appearance (0-1)
        return [freq / num_coalitions for freq in atom_frequency]
    
    elif method == 'weighted':
        # Weight by SubgraphX P scores
        total_weight = sum(p_scores)
        if total_weight > 0:
            return [score / total_weight for score in atom_weighted_scores]
        else:
            return [freq / num_coalitions for freq in atom_frequency]
    
    else:
        print(f"Unknown method: {method}. Using 'union' as default.")
        return [1 if freq > 0 else 0 for freq in atom_frequency]

def calculate_comprehensive_stats(atom_scores, atom_frequency, all_coalitions, num_atoms, method, p_scores):
    """
    Calculate comprehensive statistics for multiple coalitions.
    """
    if method in ['frequency', 'weighted']:
        num_important_atoms = sum(score > 0 for score in atom_scores)
        avg_importance = np.mean([score for score in atom_scores if score > 0]) if num_important_atoms > 0 else 0
    else:
        num_important_atoms = sum(atom_scores)
        avg_importance = 1.0 if num_important_atoms > 0 else 0
    
    # Calculate consistency metrics
    coalition_sizes = [len(coalition) for coalition in all_coalitions]
    unique_atoms = len([freq for freq in atom_frequency if freq > 0])
    
    stats = {
        'total_atoms': num_atoms,
        'important_atoms': num_important_atoms,
        'fraction_important': num_important_atoms / num_atoms,
        'percentage_important': (num_important_atoms / num_atoms) * 100,
        'method': method,
        'num_coalitions': len(all_coalitions),
        'coalition_sizes': coalition_sizes,
        'avg_coalition_size': np.mean(coalition_sizes) if coalition_sizes else 0,
        'unique_atoms_across_coalitions': unique_atoms,
        'atom_frequency': atom_frequency,
        'consistency_score': np.std(coalition_sizes) if coalition_sizes else 0,
        'avg_p_score': np.mean(p_scores) if p_scores else 0,
    }
    
    if method in ['frequency', 'weighted']:
        stats['avg_importance_score'] = avg_importance
        stats['max_importance_score'] = max(atom_scores) if atom_scores else 0
    
    return stats
