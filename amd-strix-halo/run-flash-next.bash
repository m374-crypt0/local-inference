./gufo.bash serve llm \
						--host 0.0.0.0 \
						--served-model-name flash-next \
						--model /models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
						--speculative mtp \
						--mtp-model /models/flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
