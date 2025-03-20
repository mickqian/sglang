from collections import OrderedDict

from transformers import AutoImageProcessor, AutoProcessor
from transformers.models.auto import IMAGE_PROCESSOR_MAPPING
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES,
    MODEL_NAMES_MAPPING,
)
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING,
    PROCESSOR_MAPPING_NAMES,
)


def remove_if_exists(mapping, key):
    if key in mapping:
        if isinstance(mapping, OrderedDict):
            mapping.pop(key)
            mapping.popitem(key)
            # del mapping[key]


def register_image_processor(config, image_processor):
    """
    register customized hf image processor while removing hf impl
    """
    remove_if_exists(IMAGE_PROCESSOR_MAPPING._config_mapping, config.model_type)
    remove_if_exists(IMAGE_PROCESSOR_MAPPING._model_mapping, config.model_type)
    # remove_if_exists(IMAGE_PROCESSOR_MAPPING_NAMES, config.model_type)
    # remove_if_exists(CONFIG_MAPPING_NAMES, config.model_type)
    print(IMAGE_PROCESSOR_MAPPING_NAMES)
    # print(IMAGE_PROCESSOR_MAPPING.items())
    AutoImageProcessor.register(config, None, image_processor, None)
    CONFIG_MAPPING_NAMES[config.model_type] = config.__name__
    MODEL_NAMES_MAPPING[config.model_type] = ""
    CONFIG_MAPPING[config.model_type] = config
    CONFIG_MAPPING._extra_content[config.model_type] = config


def register_processor(config, processor):
    """
    register customized hf processor while removing hf impl
    """
    print(PROCESSOR_MAPPING._config_mapping)
    print(PROCESSOR_MAPPING._model_mapping)
    remove_if_exists(PROCESSOR_MAPPING._config_mapping, config.model_type)
    remove_if_exists(PROCESSOR_MAPPING._model_mapping, config.model_type)
    remove_if_exists(PROCESSOR_MAPPING_NAMES, config.model_type)

    PROCESSOR_MAPPING._extra_content[config.model_type] = processor
    # remove_if_exists(CONFIG_MAPPING_NAMES, config.model_type)
    AutoProcessor.register(config, processor, True)
    CONFIG_MAPPING_NAMES[config.model_type] = config.__name__
    CONFIG_MAPPING[config.model_type] = config
    CONFIG_MAPPING._extra_content[config.model_type] = config
    MODEL_NAMES_MAPPING[config.model_type] = ""
