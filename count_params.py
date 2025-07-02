#!/usr/bin/env python3
"""
Standalone parameter counting script for the modded nanoGPT model.
This script initializes the model configuration and counts parameters without distributed training.
"""

import sys
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize

def count_parameters(model, print_breakdown=True):
    """
    Count the number of parameters in the model and optionally print a breakdown.
    
    Args:
        model: PyTorch model
        print_breakdown: Whether to print detailed parameter breakdown
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if print_breakdown:
        print("\n" + "="*60)
        print("MODEL PARAMETER BREAKDOWN")
        print("="*60)
        
        # Group parameters by component (top-level)
        component_params = {}
        # Group parameters by submodule (detailed)
        submodule_params = {}
        # Store all parameter names for debugging
        all_param_names = []
        
        for name, param in model.named_parameters():
            all_param_names.append(name)
            
            # Extract component name (first part before first dot)
            component = name.split('.')[0] if '.' in name else name
            
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()
            
            # Extract submodule name (everything before the last dot, or full name if no dots)
            if '.' in name:
                # Find the last parameter name (like 'weight', 'bias') and remove it
                parts = name.split('.')
                if len(parts) > 1:
                    submodule = '.'.join(parts[:-1])
                else:
                    submodule = name
            else:
                submodule = name
                
            if submodule not in submodule_params:
                submodule_params[submodule] = 0
            submodule_params[submodule] += param.numel()
        
        # Print sample parameter names for debugging
        print("SAMPLE PARAMETER NAMES (first 10):")
        print("-"*60)
        for name in sorted(all_param_names)[:10]:
            print(f"  {name}")
        if len(all_param_names) > 10:
            print(f"  ... and {len(all_param_names) - 10} more")
        
        # Create hierarchical summaries
        try:
            hierarchical_summaries = create_hierarchical_summaries(submodule_params, total_params)
        except Exception as e:
            print(f"Error creating hierarchical summaries: {e}")
            hierarchical_summaries = {}
        
        # Print high-level breakdown
        print("\nHIGH-LEVEL COMPONENTS:")
        print("-"*60)
        for component, params in sorted(component_params.items()):
            percentage = (params / total_params) * 100
            print(f"{component:<20}: {params:>12,} parameters ({percentage:>5.1f}%)")
        
        # Print hierarchical summaries
        try:
            print_hierarchical_summaries(hierarchical_summaries)
        except Exception as e:
            print(f"Error printing hierarchical summaries: {e}")
        
        # Print detailed submodule breakdown
        print("\nDETAILED SUBMODULE BREAKDOWN:")
        print("-"*60)
        for submodule, params in sorted(submodule_params.items()):
            percentage = (params / total_params) * 100
            # Truncate very long submodule names for better formatting
            display_name = submodule if len(submodule) <= 35 else f"...{submodule[-32:]}"
            print(f"{display_name:<35}: {params:>12,} parameters ({percentage:>5.1f}%)")
        
        print("-"*60)
        print(f"{'Total':<20}: {total_params:>12,} parameters (100.0%)")
        print(f"{'Trainable':<20}: {trainable_params:>12,} parameters")
        print(f"{'Memory (fp32)':<20}: {total_params * 4 / 1024**2:>8.1f} MB")
        print(f"{'Memory (bf16)':<20}: {total_params * 2 / 1024**2:>8.1f} MB")
        print("="*60)
    
    return total_params, trainable_params

def create_hierarchical_summaries(submodule_params, total_params):
    """
    Create hierarchical parameter summaries by grouping submodules with common prefixes.
    
    Args:
        submodule_params: Dict mapping submodule names to parameter counts
        total_params: Total number of parameters for percentage calculations
        
    Returns:
        Dict with hierarchical summaries organized by depth and pattern
    """
    summaries = {}
    
    # Group by different levels of hierarchy
    for submodule, params in submodule_params.items():
        parts = submodule.split('.')
        
        # Create summaries for each level of the hierarchy
        for depth in range(1, len(parts) + 1):
            prefix = '.'.join(parts[:depth])
            
            # Skip if this is already a leaf (individual submodule)
            if depth == len(parts):
                continue
                
            if prefix not in summaries:
                summaries[prefix] = {
                    'params': 0,
                    'submodules': set(),
                    'depth': depth
                }
            
            summaries[prefix]['params'] += params
            summaries[prefix]['submodules'].add(submodule)
    
    # Remove summaries that only contain one submodule (not useful for grouping)
    filtered_summaries = {k: v for k, v in summaries.items() 
                         if len(v['submodules']) > 1}
    
    return filtered_summaries

def print_hierarchical_summaries(hierarchical_summaries):
    """Print hierarchical parameter summaries organized by pattern type."""
    if not hierarchical_summaries:
        print("\nNo hierarchical summaries found.")
        return
        
    print("\nHIERARCHICAL PARAMETER SUMMARIES:")
    print("-"*60)
    
    # Group summaries by common patterns
    pattern_groups = {}
    
    for prefix, info in hierarchical_summaries.items():
        parts = prefix.split('.')
        
        # Determine the pattern type with proper bounds checking
        if len(parts) >= 3 and parts[1] == 'h' and parts[2].isdigit():
            # Transformer layer pattern (e.g., transformer.h.0, transformer.h.1)
            if len(parts) == 3:
                pattern_type = "transformer_layers"
            elif len(parts) >= 4:
                layer_component = parts[3]
                pattern_type = f"transformer_{layer_component}"
            else:
                pattern_type = "transformer_other"
        elif 'embed' in prefix.lower():
            pattern_type = "embeddings"
        elif 'head' in prefix.lower() or 'lm_head' in prefix.lower():
            pattern_type = "output_heads"
        else:
            pattern_type = "other"
            
        if pattern_type not in pattern_groups:
            pattern_groups[pattern_type] = []
        pattern_groups[pattern_type].append((prefix, info))
    
    # Print each pattern group
    total_hierarchical_params = sum(h['params'] for h in hierarchical_summaries.values())
    
    for pattern_type, items in sorted(pattern_groups.items()):
        print(f"\n{pattern_type.replace('_', ' ').title()}:")
        
        # Sort items within each group
        sorted_items = sorted(items, key=lambda x: (x[1]['depth'], x[0]))
        
        # Calculate total params for this group
        group_total_params = sum(item[1]['params'] for item in items)
        
        for prefix, info in sorted_items:
            # Calculate percentages safely
            if group_total_params > 0:
                group_percentage = (info['params'] / group_total_params) * 100
            else:
                group_percentage = 0.0
                
            if total_hierarchical_params > 0:
                total_percentage = (info['params'] / total_hierarchical_params) * 100
            else:
                total_percentage = 0.0
            
            # Count unique components
            submodule_count = len(info['submodules'])
            
            # Create display name with pattern indicators
            if pattern_type.startswith("transformer_") and "h." in prefix:
                # Show layer range for transformer components
                layer_nums = []
                for submod in info['submodules']:
                    parts = submod.split('.')
                    for i, part in enumerate(parts):
                        if part == 'h' and i + 1 < len(parts) and parts[i + 1].isdigit():
                            try:
                                layer_nums.append(int(parts[i + 1]))
                            except (ValueError, IndexError):
                                pass
                            break
                if layer_nums:
                    layer_range = f"[{min(layer_nums)}-{max(layer_nums)}]" if len(set(layer_nums)) > 1 else f"[{layer_nums[0]}]"
                    display_name = f"{prefix} {layer_range}"
                else:
                    display_name = prefix
            else:
                display_name = prefix
                
            print(f"  {display_name:<30}: {info['params']:>10,} params ({total_percentage:>4.1f}%) [{submodule_count} modules]")

def analyze_model_architecture(model, cfg):
    """Analyze and print model architecture details."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Model Type: GPT with Polynomial Mixer (PoM)")
    print(f"Vocabulary Size: {cfg.model.vocab_size:,}")
    print(f"Number of Layers: {cfg.model.n_layer}")
    print(f"Number of Heads: {cfg.model.n_head}")
    print(f"Embedding Dimension: {cfg.model.n_embd}")
    print(f"PoM Degree: {cfg.model.degree}")
    print(f"PoM Expand Factor: {cfg.model.expand}")
    print(f"Attention Mechanism: Polynomial Mixer")
    print("="*60)

def main():
    """Main function to count parameters for different configurations."""
    
    # Initialize Hydra
    with initialize(version_base=None, config_path="config"):
        # Test different configurations
        configs_to_test = [
            ("pomgpt_baseline", "POM-GPT Baseline (12 layers, 1 head)"),
            ("pomgpt_multihead", "POM-GPT Multihead (12 layers, 32 heads)"),
            ("transformers_baseline", "Transformer Baseline (standard attention)")
        ]
        
        for experiment_name, description in configs_to_test:
            try:
                print(f"\n{'='*80}")
                print(f"ANALYZING: {description}")
                print(f"Configuration: {experiment_name}")
                print(f"{'='*80}")
                
                # Compose configuration
                cfg = compose(config_name="config", overrides=[f"experiment={experiment_name}"])
                
                # Print configuration summary
                analyze_model_architecture(cfg, cfg)
                
                # Initialize model
                print(f"\nInitializing model...")
                model = instantiate(cfg.model.gpt)
                
                # Count parameters
                total_params, trainable_params = count_parameters(model, print_breakdown=True)
                
                # Calculate model size in different precisions
                print(f"\nModel Size Estimates:")
                print(f"FP32: {total_params * 4 / 1024**3:.2f} GB")
                print(f"FP16/BF16: {total_params * 2 / 1024**3:.2f} GB")
                print(f"INT8: {total_params / 1024**3:.2f} GB")
                
            except Exception as e:
                print(f"Error analyzing {experiment_name}: {e}")
                continue

if __name__ == "__main__":
    main() 