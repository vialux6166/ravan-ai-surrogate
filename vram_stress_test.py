#!/usr/bin/env python3
"""
VRAM Stress Test - Demonstrates Real Memory Savings
Shows what batch sizes are possible with different optimizations
"""

import torch
import torch.nn as nn


class VeryDeepModel(nn.Module):
    """Extremely deep model to stress VRAM"""
    
    def __init__(self, n_layers=50, hidden_dim=4096, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        return x


def find_max_batch_size(config_name, use_checkpointing=False, use_amp=False):
    """Binary search to find maximum batch size that fits in VRAM"""
    
    print(f"\n{'='*70}")
    print(f"TESTING: {config_name}")
    print(f"{'='*70}")
    
    device = torch.device('cuda')
    hidden_dim = 4096
    n_layers = 50
    
    print(f"Model: {n_layers} layers x {hidden_dim} neurons")
    print(f"Gradient Checkpointing: {'ENABLED' if use_checkpointing else 'DISABLED'}")
    print(f"Mixed Precision: {'ENABLED' if use_amp else 'DISABLED'}")
    
    # Binary search for max batch size - VERY LARGE RANGE
    low, high = 1024, 20000  # Start at 1024, go up to 20000
    max_working_batch = 0
    
    while low <= high:
        batch_size = (low + high) // 2
        
        try:
            # Clear VRAM
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create model
            model = VeryDeepModel(n_layers, hidden_dim, use_checkpointing).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            
            # Create data
            X = torch.randn(batch_size, hidden_dim, device=device)
            y = torch.randn(batch_size, hidden_dim, device=device)
            
            # Training step
            model.train()
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                scaler = torch.amp.GradScaler('cuda')
                with torch.amp.autocast('cuda'):
                    output = model(X)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
            else:
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
            
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            
            # Show progress with more detail
            free_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9) - peak_mem
            print(f"  Batch {batch_size:5d}: ✓ SUCCESS - Peak: {peak_mem:.2f} GB, Free: {free_mem:.2f} GB")
            
            max_working_batch = batch_size
            low = batch_size + 1
            
            # Cleanup
            del model, optimizer, X, y, output, loss
            if use_amp:
                del scaler
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch {batch_size:5d}: ✗ OOM - Exceeded VRAM limit")
                high = batch_size - 1
                torch.cuda.empty_cache()
            else:
                raise e
    
    # Calculate final VRAM usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        model = VeryDeepModel(n_layers, hidden_dim, use_checkpointing).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        X = torch.randn(max_working_batch, hidden_dim, device=device)
        y = torch.randn(max_working_batch, hidden_dim, device=device)
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            scaler = torch.amp.GradScaler('cuda')
            with torch.amp.autocast('cuda'):
                output = model(X)
                loss = nn.MSELoss()(output, y)
            scaler.scale(loss).backward()
        else:
            output = model(X)
            loss = nn.MSELoss()(output, y)
            loss.backward()
        
        torch.cuda.synchronize()
        final_peak = torch.cuda.max_memory_allocated() / 1e9
        
        del model, optimizer, X, y, output, loss
        if use_amp:
            del scaler
        torch.cuda.empty_cache()
        
    except:
        final_peak = 0
    
    print(f"\n✓ Maximum batch size: {max_working_batch}")
    print(f"✓ Peak VRAM at max batch: {final_peak:.2f} GB")
    
    return max_working_batch, final_peak


def main():
    """Run stress test to find maximum batch sizes"""
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print("\n" + "="*70)
    print("VRAM OPTIMIZATION STRESS TEST")
    print("="*70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\nThis test finds the MAXIMUM batch size for each configuration.")
    print("Larger batch size = better VRAM utilization!")
    
    results = {}
    
    # Test 1: Baseline
    batch, vram = find_max_batch_size(
        "Baseline (FP32, No Checkpointing)",
        use_checkpointing=False,
        use_amp=False
    )
    results['Baseline'] = {'batch': batch, 'vram': vram}
    
    # Test 2: Gradient Checkpointing
    batch, vram = find_max_batch_size(
        "Gradient Checkpointing",
        use_checkpointing=True,
        use_amp=False
    )
    results['Checkpointing'] = {'batch': batch, 'vram': vram}
    
    # Test 3: Mixed Precision
    batch, vram = find_max_batch_size(
        "Mixed Precision (FP16)",
        use_checkpointing=False,
        use_amp=True
    )
    results['Mixed Precision'] = {'batch': batch, 'vram': vram}
    
    # Test 4: Full Optimization
    batch, vram = find_max_batch_size(
        "Full Optimization (FP16 + Checkpointing)",
        use_checkpointing=True,
        use_amp=True
    )
    results['Full Optimization'] = {'batch': batch, 'vram': vram}
    
    # Print final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    baseline_batch = results['Baseline']['batch']
    baseline_vram = results['Baseline']['vram']
    
    print(f"\n{'Configuration':<30} {'Max Batch':<12} {'Peak VRAM':<12} {'Improvement':<15}")
    print("-" * 69)
    
    for name, data in results.items():
        max_batch = data['batch']
        peak_vram = data['vram']
        improvement = max_batch / baseline_batch if baseline_batch > 0 else 0
        print(f"{name:<30} {max_batch:<12d} {peak_vram:<12.2f} {improvement:<15.2f}x")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print(f"\n1. Baseline Performance:")
    print(f"   - Max batch size: {baseline_batch}")
    print(f"   - Peak VRAM: {baseline_vram:.2f} GB")
    
    checkpoint_batch = results['Checkpointing']['batch']
    checkpoint_improvement = checkpoint_batch / baseline_batch
    print(f"\n2. Gradient Checkpointing:")
    print(f"   - Max batch size: {checkpoint_batch} ({checkpoint_improvement:.2f}x improvement)")
    print(f"   - Peak VRAM: {results['Checkpointing']['vram']:.2f} GB")
    print(f"   - Saves memory by not storing intermediate activations")
    print(f"   - Recomputes them during backward pass")
    
    mixed_batch = results['Mixed Precision']['batch']
    mixed_improvement = mixed_batch / baseline_batch
    print(f"\n3. Mixed Precision (FP16):")
    print(f"   - Max batch size: {mixed_batch} ({mixed_improvement:.2f}x improvement)")
    print(f"   - Peak VRAM: {results['Mixed Precision']['vram']:.2f} GB")
    print(f"   - Uses FP16 for activations (half the memory)")
    print(f"   - Keeps FP32 master weights for stability")
    
    full_batch = results['Full Optimization']['batch']
    full_improvement = full_batch / baseline_batch
    print(f"\n4. Full Optimization (FP16 + Checkpointing):")
    print(f"   - Max batch size: {full_batch} ({full_improvement:.2f}x improvement)")
    print(f"   - Peak VRAM: {results['Full Optimization']['vram']:.2f} GB")
    print(f"   - Best of both worlds!")
    print(f"   - Can train with {full_improvement:.1f}x larger batches")
    
    # Calculate total memory savings
    vram_saved = baseline_vram - results['Full Optimization']['vram']
    vram_saved_pct = (vram_saved / baseline_vram * 100) if baseline_vram > 0 else 0
    print(f"\n5. Total Memory Savings:")
    print(f"   - Baseline uses: {baseline_vram:.2f} GB")
    print(f"   - Optimized uses: {results['Full Optimization']['vram']:.2f} GB")
    print(f"   - Saved: {vram_saved:.2f} GB ({vram_saved_pct:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ STRESS TEST COMPLETE")
    print("="*70)
    print("\nConclusion: Optimizations allow significantly larger batch sizes,")
    print("which means better GPU utilization and faster training!")


if __name__ == '__main__':
    main()
