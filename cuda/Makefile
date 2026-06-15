# RTX A6000 = sm_86 (Ampere). 학교 서버에서 nvcc 12.x 기준.
NVCC = nvcc
ARCH = -arch=sm_86
FLAGS = -O3 $(ARCH) --use_fast_math -lineinfo

STAGES = stage1_naive stage2_block stage3_fp32_curand4 stage4a_ldg stage5_multistream

all: $(STAGES)

stage1_naive: stage1_naive.cu
	$(NVCC) $(FLAGS) $< -o $@

stage2_block: stage2_block.cu
	$(NVCC) $(FLAGS) $< -o $@

stage3_fp32_curand4: stage3_fp32_curand4.cu
	$(NVCC) $(FLAGS) $< -o $@

stage4a_ldg: stage4a_ldg.cu
	$(NVCC) $(FLAGS) $< -o $@

stage5_multistream: stage5_multistream.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(STAGES) *.o

.PHONY: all clean
