#save cache files in scratch since we have bigger space here
export HF_HOME=$SCRATCH/hf_temp
#make sure you are logged in hf
huggingface-cli whoami
echo $HF_HOME

#sailor2 family
lm_eval --model hf --model_args pretrained=sail/Sailor2-1B --task fivecommonsense --output_path $SCRATCH/MS-thesis/sailor2-1b --log_samples
lm_eval --model hf --model_args pretrained=sail/Sailor2-3B --task fivecommonsense --output_path $SCRATCH/MS-thesis/sailor2-3b --log_samples
lm_eval --model hf --model_args pretrained=sail/Sailor2-8B --task fivecommonsense --output_path $SCRATCH/MS-thesis/sailor2-8b --log_samples
lm_eval --model hf --model_args pretrained=sail/Sailor2-14B --task fivecommonsense --output_path $SCRATCH/MS-thesis/sailor2-14b --log_samples




