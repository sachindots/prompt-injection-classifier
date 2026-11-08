.PHONY: train eval
train:
	uv run python -m injclf.baselines
	uv run python -m injclf.train

eval:
	uv run python -m injclf.evaluate

.PHONY: gateway
gateway:
	uv run python -m injclf.policy
