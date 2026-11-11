model=flash-next
stamp="$(date +%Y%m%d%H%M%S)"

./gufo.bash serve llm \
						--host 0.0.0.0 \
						--served-model-name $model \
						--model /models/$model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
						--speculative mtp \
						--mtp-model /models/$model/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf |
	tee "./logs/${model}-${stamp}.log"
