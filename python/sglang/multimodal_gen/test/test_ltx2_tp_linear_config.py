import ast
from pathlib import Path


def _literal_keywords(call: ast.Call) -> dict[str, object]:
    keywords: dict[str, object] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            keywords[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            continue
    return keywords


def _get_class_def(source_path: Path, class_name: str) -> ast.ClassDef:
    module = ast.parse(source_path.read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _get_class_method(
    source_path: Path, class_name: str, method_name: str
) -> ast.FunctionDef:
    target_class = _get_class_def(source_path, class_name)
    return next(
        node
        for node in target_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _get_self_call_keywords(
    source_path: Path, class_name: str, attr_name: str
) -> tuple[str, dict[str, object]]:
    init_fn = _get_class_method(source_path, class_name, "__init__")
    for node in init_fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == attr_name
            and isinstance(node.value, ast.Call)
        ):
            func = node.value.func
            func_name = func.id if isinstance(func, ast.Name) else None
            keywords = _literal_keywords(node.value)
            return func_name, keywords
    raise AssertionError(
        f"Did not find self.{attr_name} assignment in {class_name} from {source_path}"
    )


def _get_self_sequential_first_call_keywords(
    source_path: Path, class_name: str, attr_name: str
) -> tuple[str, dict[str, object]]:
    init_fn = _get_class_method(source_path, class_name, "__init__")
    for node in init_fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            not isinstance(target, ast.Attribute)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "self"
            or target.attr != attr_name
            or not isinstance(node.value, ast.Call)
        ):
            continue
        if not isinstance(node.value.func, ast.Attribute):
            continue
        if node.value.func.attr != "Sequential" or not node.value.args:
            continue
        first_call = node.value.args[0]
        if not isinstance(first_call, ast.Call):
            continue
        func = first_call.func
        func_name = func.id if isinstance(func, ast.Name) else None
        return func_name, _literal_keywords(first_call)
    raise AssertionError(
        f"Did not find self.{attr_name} Sequential assignment in {class_name} from {source_path}"
    )


def _get_module_function(source_path: Path, function_name: str) -> ast.FunctionDef:
    module = ast.parse(source_path.read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_ltx2_feed_forward_uses_tp_friendly_linear_layers():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )

    proj_in_func, proj_in_keywords = _get_self_call_keywords(
        source_path, "LTX2FeedForward", "proj_in"
    )
    proj_out_func, proj_out_keywords = _get_self_call_keywords(
        source_path, "LTX2FeedForward", "proj_out"
    )

    assert proj_in_func == "ColumnParallelLinear"
    assert proj_in_keywords["gather_output"] is False
    assert proj_out_func == "RowParallelLinear"
    assert proj_out_keywords["input_is_parallel"] is True
    assert proj_out_keywords["accumulate_in_fp32"] is True


def test_ltx2_text_projection_uses_tp_friendly_linear_layers():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )

    linear_1_func, linear_1_keywords = _get_self_call_keywords(
        source_path, "LTX2TextProjection", "linear_1"
    )
    linear_2_func, linear_2_keywords = _get_self_call_keywords(
        source_path, "LTX2TextProjection", "linear_2"
    )

    assert linear_1_func == "ColumnParallelLinear"
    assert linear_1_keywords["gather_output"] is False
    assert linear_1_keywords["accumulate_in_fp32"] is True
    assert linear_2_func == "RowParallelLinear"
    assert linear_2_keywords["input_is_parallel"] is True
    assert linear_2_keywords["accumulate_in_fp32"] is True


def test_ltx2_timestep_embedder_keeps_outputs_gathered():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )

    linear_1_func, linear_1_keywords = _get_self_call_keywords(
        source_path, "LTX2TimestepEmbedder", "linear_1"
    )
    linear_2_func, linear_2_keywords = _get_self_call_keywords(
        source_path, "LTX2TimestepEmbedder", "linear_2"
    )

    assert linear_1_func == "ColumnParallelLinear"
    assert linear_1_keywords["gather_output"] is True
    assert linear_2_func == "ColumnParallelLinear"
    assert linear_2_keywords["gather_output"] is True


def test_ltx2_adaln_single_uses_fp32_column_parallel_accumulation():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )

    linear_func, linear_keywords = _get_self_call_keywords(
        source_path, "LTX2AdaLayerNormSingle", "linear"
    )

    assert linear_func == "ColumnParallelLinear"
    assert linear_keywords["gather_output"] is True
    assert linear_keywords["accumulate_in_fp32"] is True


def test_ltx2_final_output_heads_use_fp32_column_parallel_accumulation():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )

    proj_out_func, proj_out_keywords = _get_self_call_keywords(
        source_path, "LTX2VideoTransformer3DModel", "proj_out"
    )
    audio_proj_out_func, audio_proj_out_keywords = _get_self_call_keywords(
        source_path, "LTX2VideoTransformer3DModel", "audio_proj_out"
    )

    assert proj_out_func == "ColumnParallelLinear"
    assert proj_out_keywords["gather_output"] is True
    assert proj_out_keywords["accumulate_in_fp32"] is True
    assert audio_proj_out_func == "ColumnParallelLinear"
    assert audio_proj_out_keywords["gather_output"] is True
    assert audio_proj_out_keywords["accumulate_in_fp32"] is True


def test_ltx2_attention_output_projection_uses_fp32_row_parallel_accumulation():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )

    to_q_func, to_q_keywords = _get_self_call_keywords(source_path, "LTX2Attention", "to_q")
    to_k_func, to_k_keywords = _get_self_call_keywords(source_path, "LTX2Attention", "to_k")
    to_v_func, to_v_keywords = _get_self_call_keywords(source_path, "LTX2Attention", "to_v")
    to_out_func, to_out_keywords = _get_self_sequential_first_call_keywords(
        source_path, "LTX2Attention", "to_out"
    )

    assert to_q_func == "ColumnParallelLinear"
    assert to_q_keywords["gather_output"] is False
    assert to_q_keywords["accumulate_in_fp32"] is True
    assert to_k_func == "ColumnParallelLinear"
    assert to_k_keywords["gather_output"] is False
    assert to_k_keywords["accumulate_in_fp32"] is True
    assert to_v_func == "ColumnParallelLinear"
    assert to_v_keywords["gather_output"] is False
    assert to_v_keywords["accumulate_in_fp32"] is True
    assert to_out_func == "RowParallelLinear"
    assert to_out_keywords["input_is_parallel"] is True
    assert to_out_keywords["accumulate_in_fp32"] is True


def test_ltx2_tp_rms_norm_casts_after_affine_weight():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )
    forward_fn = _get_class_method(source_path, "LTX2TPRMSNormAcrossHeads", "forward")

    assign_names = [
        node.targets[0].id
        for node in forward_fn.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    assert "y_fp32" in assign_names

    affine_assignments = [
        node
        for node in forward_fn.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "y_fp32"
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Mult)
        and isinstance(node.value.right, ast.Call)
        and isinstance(node.value.right.func, ast.Attribute)
        and isinstance(node.value.right.func.value, ast.Attribute)
        and isinstance(node.value.right.func.value.value, ast.Name)
        and node.value.right.func.value.value.id == "self"
        and node.value.right.func.value.attr == "weight"
        and node.value.right.func.attr == "float"
    ]
    assert affine_assignments

    return_stmt = next(
        node for node in forward_fn.body if isinstance(node, ast.Return)
    )
    assert isinstance(return_stmt.value, ast.Call)
    assert isinstance(return_stmt.value.func, ast.Attribute)
    assert isinstance(return_stmt.value.func.value, ast.Name)
    assert return_stmt.value.func.value.id == "y_fp32"
    assert return_stmt.value.func.attr == "to"


def test_ltx2_attention_exposes_qk_norm_and_pre_to_out_trace_points():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )
    forward_fn = _get_class_method(source_path, "LTX2Attention", "forward")

    trace_names = {
        node.args[0].value
        for node in ast.walk(forward_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "maybe_trace"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    assert {
        "input",
        "q_proj",
        "q_proj_reconstructed_full",
        "k_proj",
        "k_proj_reconstructed_full",
        "v_proj",
        "v_proj_reconstructed_full",
        "q_norm",
        "k_norm",
        "gate_logits",
        "pre_to_out",
        "to_out_reconstructed_full",
    } <= trace_names


def test_ltx2_attention_gathers_trace_tensors_before_export():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )
    forward_fn = _get_class_method(source_path, "LTX2Attention", "forward")

    helper_calls = [
        node
        for node in ast.walk(forward_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_gather_trace_tensor_for_tp"
    ]

    assert helper_calls, "LTX2Attention.forward should gather TP trace tensors before export"


def test_ltx2_model_forward_exposes_pre_block_trace_points():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )
    forward_fn = _get_class_method(source_path, "LTX2VideoTransformer3DModel", "forward")

    trace_names = {
        node.args[0].value
        for node in ast.walk(forward_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "maybe_record_model_trace"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    assert {
        "video_patchify_input",
        "video_patchify_proj",
        "video_patchify_proj_reconstructed_full",
        "video_embedded_timestep",
        "video_temb",
        "video_caption_projection",
        "video_norm_out",
        "video_proj_out_input",
        "video_proj_out",
    } <= trace_names


def test_ltx2_transformer_block_exposes_video_attn1_modulation_trace_points():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "models" / "dits" / "ltx_2.py"
    )
    forward_fn = _get_class_method(source_path, "LTX2TransformerBlock", "forward")

    trace_names = {
        node.args[1].value
        for node in ast.walk(forward_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_ltx2_record_trace"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }

    assert {
        "video_attn1_rmsnorm",
        "video_attn1_shift",
        "video_attn1_scale",
        "video_attn1_gate",
        "video_post_attn1_hidden_states",
        "video_post_attn2_hidden_states",
    } <= trace_names


def test_fp32_accum_helpers_do_not_gate_on_tp_size():
    source_path = (
        Path(__file__).resolve().parents[1] / "runtime" / "layers" / "linear.py"
    )

    for function_name in (
        "should_use_fp32_row_parallel_accum",
        "should_use_fp32_column_parallel_accum",
    ):
        function_def = _get_module_function(source_path, function_name)
        tp_size_guards = [
            node
            for node in ast.walk(function_def)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and isinstance(node.left.value, ast.Name)
            and node.left.value.id == "layer"
            and node.left.attr == "tp_size"
        ]
        assert not tp_size_guards, (
            f"{function_name} should not special-case tp_size; "
            "single-gpu and TP must honor accumulate_in_fp32 consistently."
        )
