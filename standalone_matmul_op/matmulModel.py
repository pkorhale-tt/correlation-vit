import sys
import torch
import ttnn

def main():
    # Usage: python matmul_wh_lofi.py 6272 1024 256
    if len(sys.argv) != 4:
        print("Usage: python matmul_wh_lofi.py M K N")
        sys.exit(1)

    try:
        M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    except ValueError:
        print("Error: M, K, N must be integers.")
        sys.exit(1)

    if M <= 0 or K <= 0 or N <= 0:
        print("Error: M, K, N must be positive.")
        sys.exit(1)

    print(f"\nOpening device 0...")
    device = ttnn.open_device(device_id=0)

    try:
        print(f"Creating A[{M}, {K}] and B[{K}, {N}] (BFLOAT8_B target)...")
        a_torch = torch.randn((M, K), dtype=torch.bfloat16)
        b_torch = torch.randn((K, N), dtype=torch.bfloat16)

        print("Moving tensors to L1 on device as BFLOAT8_B...")
        a_tt = ttnn.to_device(
            ttnn.from_torch(
                a_torch,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        b_tt = ttnn.to_device(
            ttnn.from_torch(
                b_torch,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Explicit LoFi compute kernel config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        print("Requested math_fidelity:", compute_config.math_fidelity)

        # Run matmul 10 times
        for i in range(10):
            print(f"Run {i+1}/10: BFLOAT8_B LoFi matmul...")
            output_tensor = ttnn.matmul(
                a_tt,
                b_tt,
                compute_kernel_config=compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        result_torch = ttnn.to_torch(output_tensor)
        print("\nMatmul successful (BFLOAT8_B + requested LoFi).")
        print(f"A shape: {a_torch.shape}")
        print(f"B shape: {b_torch.shape}")
        print(f"C = A @ B shape: {result_torch.shape}")

    finally:
        print("Closing device...")
        ttnn.close_device(device)

if __name__ == "__main__":
    main()
