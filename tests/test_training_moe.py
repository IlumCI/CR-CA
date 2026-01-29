from training.finetune import _resolve_model_info


def test_resolve_model_info_switch_is_seq2seq() -> None:
    info = _resolve_model_info("google/switch-base-8")
    assert info["arch"] == "seq2seq"
    assert info["moe"] is True


def test_resolve_model_info_qwen_is_causal() -> None:
    info = _resolve_model_info("Qwen/Qwen2.5-1.5B-Instruct")
    assert info["arch"] == "causal"
    assert info["moe"] is False
