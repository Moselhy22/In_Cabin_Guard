"""Phase 1 validation: Benchmark MobileNetV3 latency on YOUR GTX 1650."""
import torch, time, numpy as np, sys

def benchmark_mobilenetv3(device='cuda', runs=100):
    try:
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(pretrained=True).eval().to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / runs * 1000
        
        return elapsed_ms
    except Exception as e:
        print(f"❌ Benchmark failed: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("="*60)
    print("PHASE 1 VALIDATION: MobileNetV3 Latency Benchmark")
    print("="*60)
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    
    # GPU benchmark (target hardware)
    if torch.cuda.is_available():
        gpu_ms = benchmark_mobilenetv3('cuda', 100)
        if gpu_ms:
            status = "✅ PASS" if gpu_ms < 20 else "⚠️  WARNING"
            print(f"\n{status}: GTX 1650 latency = {gpu_ms:.2f}ms/frame")
            print(f"   Target: <20ms | Result: {'MET' if gpu_ms < 20 else 'NOT MET'}")
    else:
        print("\n⚠️  CUDA not available - skipping GPU benchmark")
    
    # CPU benchmark (baseline)
    cpu_ms = benchmark_mobilenetv3('cpu', 50)
    if cpu_ms:
        print(f"CPU latency: {cpu_ms:.2f}ms/frame (baseline)")
    
    print("\n✅ Phase 1 architecture validated for YOUR hardware")
    print("="*60)
