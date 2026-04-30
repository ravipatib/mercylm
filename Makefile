.PHONY: prepare train chat eval export clean

prepare:
	python -m tlha prepare

train:
	python -m tlha train

chat:
	python -m tlha chat

eval:
	python -m tlha eval

export:
	python tools/export_to_hf.py \
		--model-repo $(MODEL_REPO) \
		--data-repo  $(DATA_REPO)  \
		--token      $(HF_TOKEN)

clean:
	rm -rf data/ checkpoints/ hf_export/ __pycache__/ tlha/__pycache__/
