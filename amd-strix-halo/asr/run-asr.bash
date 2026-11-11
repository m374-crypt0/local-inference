model=asr
stamp="$(date +%Y%m%d%H%M%S)"

../gufo.bash serve asr \
						 --host 0.0.0.0 \
						 --port 8081 \
						 --served-model-name $model \
						 --model /models/$model |
	tee "./logs/${model}-${stamp}.log"
