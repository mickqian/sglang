from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from sglang.utils import logger


class Qwen2_5OmniVisionEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5OmniThinkerVision`]. It is used to instantiate a
    Qwen2.5-VL vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2.5-VL
    architecture.

    e.g. [Qwen/Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of layers (depth) in the model.
        embed_dim (`int`, *optional*, defaults to 1280):
            Dimensionality of the embeddings.
        hidden_size (`int`, *optional*, defaults to 3584):
            The size of the hidden layers.
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function used in the model. Supported options include `"quick_gelu"` and others as applicable.
        mlp_ratio (`float`, *optional*, defaults to 4):
            The ratio used to determine the size of the MLP (Multi-Layer Perceptron) hidden layer.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        patch_size (`int`, *optional*, defaults to 14):
            The size of the patches extracted from the input.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The size used for patches along the temporal dimension.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the initializer used for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen2_5OmniVisionEncoderConfig, Qwen2_5OmniVisionEncoder

    >>> # Initializing a Qwen2_5OmniVisionEncoderConfig
    >>> configuration = Qwen2_5OmniVisionEncoderConfig()

    >>> # Initializing a Qwen2_5OmniVisionEncoder (with random weights)
    >>> model = Qwen2_5OmniVisionEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_omni_vision_encoder"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        embed_dim=1280,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        init_std=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.init_std = init_std


class Qwen2_5OmniAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5OmniAudioEncoder`]. It is used to instantiate a
    Qwen2.5-Omni-Thinker audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.

    e.g. [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Qwen2_5OmniProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        n_window (`int`, *optional*, defaults to 100):
            The chunk for conv and flash attn in AudioEncoder.
        output_dim (`int`, *optional*, defaults to 3584):
            The output dimention of AudioEncoder.

    Example:

    ```python
    >>> from transformers import Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniAudioEncoder

    >>> # Initializing a Qwen2_5OmniAudioEncoderConfig
    >>> configuration = Qwen2_5OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2_5OmniAudioEncoder (with random weights)
    >>> model = Qwen2_5OmniAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_omni_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim


class Qwen2_5OmniTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5OmniThinkerForConditionalGeneration`]. It is used to instantiate an
    Qwen2.5-Omni-Thinker model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2.5-Omni-Thinker.

    e.g. [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the QwenOmni model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2VLModel`]
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 32768):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_scaling (`Dict`, *optional*, defaults to `default`):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniVisionEncoderConfig

    >>> # Initializing a Qwen2_5OmniAudioEncoder config
    >>> audio_config = Qwen2_5OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2_5OmniVisionEncoder config
    >>> vision_config = Qwen2_5OmniVisionEncoderConfig()

    >>> # Initializing a Qwen2.5OmniThinker configuration
    >>> configuration = Qwen2_5OmniThinkerConfig(audio_config, vision_config)

    >>> # Initializing a model from the Qwen-Omni style configuration
    >>> model = Qwen2_5OmniThinkerForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_omni_text"
    is_composition = False

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-06,
        use_cache=True,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        attention_dropout=0.0,
        rope_scaling="default",
        init_std=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.init_std = init_std

        super().__init__(**kwargs)


class Qwen2_5OmniThinkerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5OmniThinkerForConditionalGeneration`]. It is used to instantiate an
    Qwen2.5-Omni-Thinker model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2.5-Omni-Thinker.

    e.g. [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`dict`,  *optional*):
            The config dictionary of the audio backbone.
        vision_config (`dict`, *optional*):
            The config dictionary of the vision backbone.
        text_config (`dict`, *optional*):
            The config dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.
        image_token_index (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_index (`int`, *optional*, defaults to 151656):
            The video token index to encode the video prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        position_id_per_seconds (`int`, *optional*, defaults to 25):
            The increment of position id per second.
        seconds_per_chunk (`int`, *optional*, defaults to 2):
            The duration in seconds of the chunk of audio and video data.
        audio_start_token_id (`int`, *optional*, defaults to 151647):
            The audio start token index to encode the audio prompt.
        audio_end_token_id (`int`, *optional*, defaults to 151648):
            The audio end token index to encode the audio prompt.
        user_token_id (`int, *optional*, defaults to 872):
            The user token index to encode the user token.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniVisionEncoderConfig

    >>> # Initializing a Qwen2_5OmniAudioEncoder config
    >>> audio_config = Qwen2_5OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2_5OmniVisionEncoder config
    >>> vision_config = Qwen2_5OmniVisionEncoderConfig()

    >>> # Initializing a Qwen2_5OmniTextConfig config
    >>> text_config = Qwen2_5OmniTextConfig()

    >>> # Initializing a Qwen2.5OmniThinker configuration
    >>> configuration = Qwen2_5OmniThinkerConfig(audio_config, vision_config, text_config)

    >>> # Initializing a model from the Qwen-Omni style configuration
    >>> model = Qwen2_5OmniThinkerForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_omni_thinker"
    sub_configs = {
        "audio_config": Qwen2_5OmniAudioEncoderConfig,
        "vision_config": Qwen2_5OmniVisionEncoderConfig,
        "text_config": Qwen2_5OmniTextConfig,
    }
    is_composition = True

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_index=151646,
        image_token_index=151655,
        video_token_index=151656,
        tie_word_embeddings=False,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
        user_token_id=872,
        init_std=0.02,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        # 2025.02.20 the add
        self.user_token_id = user_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.seconds_per_chunk = seconds_per_chunk
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.init_std = init_std

        if isinstance(vision_config, dict):
            vision_config = Qwen2_5OmniVisionEncoderConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen2_5OmniVisionEncoderConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Qwen2_5OmniAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen2_5OmniAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen2_5OmniTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen2_5OmniTextConfig()
        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen2_5OmniTalkerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5OmniTalkerForConditionalGeneration`]. It is used to instantiate an
    Qwen2.5-Omni-Talker model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2.5-Omni-Thinker.

    e.g. [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.
        image_token_index (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_index (`int`, *optional*, defaults to 151656):
            The video token index to encode the video prompt.
        vocab_size (`int`, *optional*, defaults to 8448):
            Vocabulary size of the QwenOmni model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2VLModel`]
        tts_text_start_token_id (`int`, *optional*, defaults to 151860):
            The tts text start token index to encode the start of tts text.
        tts_text_end_token_id (`int`, *optional*, defaults to 151861):
            The tts text end token index to encode the end of tts text.
        tts_text_pad_token_id (`int`, *optional*, defaults to 151859):
            The tts text pad token index to encode the pad of tts text.
        tts_codec_start_token_id (`int`, *optional*, defaults to 8293):
            The tts codec start token index to encode the start of tts codec.
        tts_codec_end_token_id (`int`, *optional*, defaults to 8294):
            The tts codec end token index to encode the end of tts codec.
        tts_codec_pad_token_id (`int`, *optional*, defaults to 8292):
            The tts codec pad token index to encode the pad of tts codec.
        tts_codec_mask_token_id (`int`, *optional*, defaults to 8296):
            The tts codec mask token index to encode the mask of tts codec.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The tts vision start token index to encode the start of vision.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The tts vision end token index to encode the end of vision.
        embedding_size (`int`, *optional*, defaults to 3584):
            Dimension of the embedding representations.
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`int`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of each attention head.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 32768):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        position_id_per_seconds (`int`, *optional*, defaults to 25):
            The increment of position id per second.
        seconds_per_chunk (`int`, *optional*, defaults to 2):
            The duration in seconds of the chunk of audio and video data.
        audio_start_token_id (`int`, *optional*, defaults to 151647):
            The audio start token index to encode the audio prompt.
        audio_end_token_id (`int`, *optional*, defaults to 151648):
            The audio end token index to encode the audio prompt.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.

    Example:

    ```python
    >>> from transformers import Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniVisionEncoderConfig

    >>> # Initializing a Qwen2_5OmniAudioEncoder config
    >>> audio_config = Qwen2_5OmniAudioEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing a Qwen2_5Omni configuration
    >>> configuration = Qwen2_5OmniThinkerConfig(audio_config, text_config)

    >>> # Initializing a model from the qwen2-audio style configuration
    >>> model = Qwen2_5OmniTalkerForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_omni_talker"
    is_composition = False

    def __init__(
        self,
        audio_token_index=151646,
        image_token_index=151655,
        video_token_index=151656,
        vocab_size=8448,
        tts_text_start_token_id=151860,
        tts_text_end_token_id=151861,
        tts_text_pad_token_id=151859,
        tts_codec_start_token_id=8293,
        tts_codec_end_token_id=8294,
        tts_codec_pad_token_id=8292,
        tts_codec_mask_token_id=8296,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        embedding_size=3584,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        head_dim=128,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        attention_dropout=0.0,
        rope_scaling=None,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
        init_std=0.02,
        spatial_merge_size=2,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index

        self.tts_text_start_token_id = tts_text_start_token_id
        self.tts_text_end_token_id = tts_text_end_token_id
        self.tts_text_pad_token_id = tts_text_pad_token_id
        self.tts_codec_start_token_id = tts_codec_start_token_id
        self.tts_codec_end_token_id = tts_codec_end_token_id
        self.tts_codec_pad_token_id = tts_codec_pad_token_id

        self.tts_codec_mask_token_id = tts_codec_mask_token_id

        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.embedding_size = embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.position_id_per_seconds = position_id_per_seconds  # zf
        self.seconds_per_chunk = seconds_per_chunk  # zf
        self.audio_start_token_id = audio_start_token_id  # zf
        self.audio_end_token_id = audio_end_token_id  # zf

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does defeault RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        self.init_std = init_std
        self.spatial_merge_size = spatial_merge_size

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen2_5OmniDiTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Qwen2_5OmniToken2WavDiT used in the Qwen2.5-Omni-Token2Wav model.
    It defines the architecture of the DiT model, which is used for generating mel-spectrograms from tokens.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            The dimension of the model.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            The number of transformer blocks in the DiT model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of attention heads in each transformer block.
        ff_mult (`int`, *optional*, defaults to 2):
            The multiplier for the feedforward layer in each transformer block.
        emb_dim (`int`, *optional*, defaults to 512):
            The dimension of the embedding layer.
        head_dim (`int`, *optional*, defaults to 64):
            The dimension of each attention head.
        repeats (`int`, *optional*, defaults to 2):
            The number of times the codec embeddings are repeated.
        num_embeds (`int`, *optional*, defaults to 8193):
            The number of unique embeddings in the codec.
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the transformer blocks.

        enc_emb_dim (`int`, *optional*, defaults to 192):
            The dimension of the pre-trained speaker embedding.
        enc_dim (`int`, *optional*, defaults to 128):
            The dimension of the encoder output.
        enc_lin_neurons (`int`, *optional*, defaults to 192):
            Number of neurons in linear layers.
        enc_channels (`List[int]`, *optional*, defaults to `[256, 256, 256, 256, 768]`):
            A list of output channels for each TDNN/SERes2Net layer in the encoder.
        enc_kernel_sizes (`List[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            A list of kernel sizes for each layer in the encoder.
        enc_dilations (`List[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            A list of dilations for each layer in the encoder.
        enc_attention_channels (`int`, *optional*, defaults to 64):
            The number of attention channels in the SEBlock.
        enc_res2net_scale (`int`, *optional*, defaults to 2):
            The scale of the Res2Net block in the encoder.
        enc_se_channels (`int`, *optional*, defaults to 64):
            The number of output channels after squeeze in the SEBlock.
    """

    model_type = "qwen2_5_omni_dit"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=22,
        num_attention_heads=16,
        ff_mult=2,
        emb_dim=512,
        head_dim=64,
        rope_theta=10000.0,
        max_position_embeddings=32768,
        block_size=24,
        look_ahead_layers=[10],
        look_backward_layers=[0, 20],
        repeats=2,
        num_embeds=8193,
        mel_dim=80,
        dropout=0.1,
        enc_emb_dim=192,
        enc_dim=128,
        enc_lin_neurons=192,
        enc_channels=[256, 256, 256, 256, 768],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=64,
        enc_res2net_scale=2,
        enc_se_channels=64,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ff_mult = ff_mult
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.block_size = block_size
        self.look_ahead_layers = look_ahead_layers
        self.look_backward_layers = look_backward_layers
        self.repeats = repeats
        self.num_embeds = num_embeds
        self.mel_dim = mel_dim
        self.dropout = dropout
        self.enc_emb_dim = enc_emb_dim
        self.enc_dim = enc_dim
        self.enc_lin_neurons = enc_lin_neurons
        self.enc_channels = enc_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_dilations = enc_dilations
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        super().__init__(**kwargs)


class Qwen2_5OmniBigVGANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Qwen2_5OmniToken2WavBigVGAN module used in the Qwen2.5-Omni-Token2Wav model.
    It defines the architecture of the BigVGAN model, which is used for converting mel-spectrograms to waveforms.

    Args:
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            The number of channels in the initial upsampling layer.
        resblock_kernel_sizes (`List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A list of kernel sizes for each residual block.
        resblock_dilation_sizes (`List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A list of dilation sizes for each residual block.
        upsample_rates (`List[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
            A list of upsampling rates for each upsampling layer.
        upsample_kernel_sizes (`List[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
            A list of kernel sizes for each upsampling layer.
    """

    model_type = "qwen2_5_omni_bigvgan"

    def __init__(
        self,
        mel_dim=80,
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[5, 3, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 7, 4, 4, 4, 4],
        **kwargs,
    ):
        self.mel_dim = mel_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        super().__init__(**kwargs)


class Qwen2_5OmniToken2WavConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [Qwen2_5OmniToken2WavModel].
    It is used to instantiate the Qwen2.5-Omni-Token2Wav model which combines a Diffusion Transformer (DiT) for mel-spectrogram generation with a BigVGAN model for waveform synthesis. The configuration contains sub-configurations for both components.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        dit_config ([`DiT_Args`], *optional*):
            Configuration class for the Diffusion Transformer (DiT) module responsible for generating mel-spectrograms.
        bigvgan_config ([`BigVGAN_Args`], *optional*):
            Configuration class for the BigVGAN module responsible for converting mel-spectrograms to waveforms.
    Example:

    ```python
    >>> from transformers import Qwen2_5OmniToken2WavModel, DiT_Args, BigVGAN_Args

    >>> # Initialize DiT configuration
    >>> dit_config = DiT_Args(
    ...     dim=1024,
    ...     depth=22,
    ...     heads=16,
    ...     ff_mult=2
    ... )

    >>> # Initialize BigVGAN configuration
    >>> bigvgan_config = BigVGAN_Args(
    ...     mel_dim=80,
    ...     upsample_rates=[5,3,2,2,2,2]
    ... )

    >>> # Initialize main configuration
    >>> config = Qwen2_5OmniToken2WavConfig(dit_config, bigvgan_config)

    >>> # Initialize model with config
    >>> model = Qwen2_5OmniToken2Wav(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """

    model_type = "qwen2_5_omni_token2wav"
    sub_configs = {
        "dit_config": Qwen2_5OmniDiTConfig,
        "bigvgan_config": Qwen2_5OmniBigVGANConfig,
    }
    is_composition = True

    def __init__(self, dit_config=None, bigvgan_config=None, **kwargs):
        if dit_config is None:
            dit_config = {}
        if bigvgan_config is None:
            bigvgan_config = {}
        self.dit_config = Qwen2_5OmniDiTConfig(**dit_config)
        self.bigvgan_config = Qwen2_5OmniBigVGANConfig(**bigvgan_config)
        super().__init__(**kwargs)


class Qwen2_5OmniConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen2_5OmniModel`]. It is used to instantiate a Qwen2.5Omni
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Qwen2.5-Omni]() architecture.
    #TODO: add link

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        thinker_config (`dict`, *optional*): Configuration of the underlying thinker sub-model.
        talker_config (`dict`, *optional*): Configuration of the underlying talker sub-model.
        token2wav_config (`dict`, *optional*): Configuration of the underlying codec sub-model.
        enable_audio_output (`bool`, *optional*, defaults to `True`): Whether enabel audio output and load talker and token2wav module.

    Example:

    ```python
    >>> from transformers import (
    ...     Qwen2_5OmniThinkerConfig,
    ...     Qwen2_5OmniTalkerConfig,
    ...     Qwen2_5OmniToken2WavConfig,
    ...     Qwen2_5OmniModel,
    ...     Qwen2_5OmniConfig,
    ... )

    >>> # Initializing sub-modules configurations.
    >>> thinker_config = Qwen2_5OmniThinkerConfig()
    >>> talker_config = Qwen2_5OmniTalkerConfig()
    >>> token2wav_config = Qwen2_5OmniToken2WavConfig()


    >>> # Initializing a module style configuration
    >>> configuration = Qwen2_5OmniConfig.from_sub_model_configs(
    ...     thinker_config, talker_config, token2wav_config
    ... )

    >>> # Initializing a model (with random weights)
    >>> model = Qwen2_5OmniModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "qwen2_5_omni"
    sub_configs = {
        "thinker_config": Qwen2_5OmniThinkerConfig,
        "talker_config": Qwen2_5OmniTalkerConfig,
        "token2wav_config": Qwen2_5OmniToken2WavConfig,
    }
    is_composition = True

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        token2wav_config=None,
        enable_audio_output: bool = True,
        **kwargs,
    ):
        if thinker_config is None:
            thinker_config = {}
            logger.info(
                "thinker_config is None. Initializing thinker model with default values"
            )

        if talker_config is None:
            talker_config = {}
            logger.info(
                "talker_config is None. Initializing talker model with default values"
            )

        if token2wav_config is None:
            token2wav_config = {}
            logger.info(
                "token2wav_config is None. Initializing token2wav model with default values"
            )

        self.thinker_config = Qwen2_5OmniThinkerConfig(**thinker_config)
        self.talker_config = Qwen2_5OmniTalkerConfig(**talker_config)
        self.token2wav_config = Qwen2_5OmniToken2WavConfig(**token2wav_config)
        self.enable_audio_output = enable_audio_output

        super().__init__(**kwargs)

    @classmethod
    def from_sub_model_configs(
        cls,
        thinker_config: Qwen2_5OmniThinkerConfig,
        talker_config: Qwen2_5OmniTalkerConfig,
        token2wav_config: Qwen2_5OmniToken2WavConfig,
        enable_audio_output: bool = True,
        **kwargs,
    ):
        r"""
        Instantiate a [`Qwen2_5OmniConfig`] (or a derived class) from sub-models configuration.

        Returns:
            [`Qwen2_5OmniConfig`]: An instance of a configuration object
        """
        return cls(
            thinker_config=thinker_config.to_dict(),
            talker_config=talker_config.to_dict(),
            token2wav_config=token2wav_config.to_dict(),
            enable_audio_output=enable_audio_output,
            **kwargs,
        )
