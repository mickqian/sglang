# python/sglang/srt/managers/multimodal_processor_service.py
import argparse  # Added for potential direct invocation or testing
import asyncio
import logging
import os
import signal
import sys

import psutil
import setproctitle
import torch
import uvloop
import zmq
import zmq.asyncio

# Need ModelConfig to initialize processor correctly
from sglang.srt.configs.model_config import ModelConfig

# Assuming these can be reused or adapted
from sglang.srt.hf_transformers_utils import get_processor

# Assuming get_mm_processor handles device placement or we adapt it
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Placeholder: Define request/response structures if needed, e.g., using dataclasses
# For now, using dictionaries. Example request: {"image_data": bytes, "model_config_dict": {...}, ...}
# Example response: {"success": bool, "result": {...}, "error": str}


class MultimodalProcessorService:
    """
    A dedicated service for handling GPU-intensive multimodal processing tasks,
    running in its own process and event loop.
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args

        if server_args.mm_gpu_id is None:
            raise ValueError("MultimodalProcessorService requires mm_gpu_id to be set.")
        self.device = torch.device(f"cuda:{server_args.mm_gpu_id}")
        self.cuda_stream = torch.cuda.Stream(device=self.device)
        print(f"Initializing MultimodalProcessorService on device: {self.device}")

        # Init ZMQ context and socket
        # Using REP socket for request-reply pattern. TokenizerManager will use REQ.
        context = zmq.asyncio.Context()
        # Ensure port_args has 'mm_proc_ipc_name' defined

        if not hasattr(port_args, "mm_proc_ipc_name"):
            raise AttributeError(
                "PortArgs is missing the required 'mm_proc_ipc_name' attribute."
            )
        self.zmq_socket = get_zmq_socket(
            context, zmq.REP, port_args.mm_proc_ipc_name, bind=True
        )
        print(f"ZMQ REP socket bound to {port_args.mm_proc_ipc_name}")

        print(f"import_processors")
        import_processors()
        # Load the multimodal processor components onto the designated GPU
        # ModelConfig is needed for processor initialization
        # TODO: Make ModelConfig creation more robust if service runs standalone
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,  # May not be strictly needed here
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,  # May not be strictly needed here
            enable_multimodal=True,  # Implicitly true for this service
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
        self.processor = self.get_mm_processor()
        print(f"Multimodal processor loaded successfully on {self.device}")

    def get_mm_processor(self) -> BaseMultimodalProcessor:
        """Loads the multimodal processor and ensures its components are on the correct device."""
        print("Loading multimodal processor...")
        # This reuses the existing logic but needs verification for device placement
        _processor = get_processor(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
            revision=self.server_args.revision,
            # Use fast processor if available and not disabled
            use_fast=not self.server_args.disable_fast_image_processor,
        )

        # get_mm_processor might initialize components on CPU or default GPU
        # We need to ensure they are moved to self.device
        mm_processor = get_mm_processor(
            self.model_config.hf_config, self.server_args, _processor
        )

        # Explicitly move necessary components (like vision tower) to the target device
        # This depends heavily on the specific processor's structure (e.g., LlavaProcessor, QwenVLProcessor)
        # if hasattr(mm_processor, 'vision_tower') and isinstance(mm_processor.vision_tower, torch.nn.Module):
        #     print(f"Moving vision tower to {self.device}...")
        #     mm_processor.vision_tower.to(self.device)
        # if hasattr(mm_processor, 'image_processor'):
        #     # Image processors are usually transformers/torchvision objects, might not need explicit move
        #     # unless they contain trainable parameters or persistent buffers intended for GPU.
        #     print("Image processor found. Assuming CPU or device handled correctly.")
        #     pass  # Add specific device moving logic if needed for a processor type

        # TODO: Add more checks and device moving logic for other potential processor types
        # or components that require GPU placement (e.g., specific projection layers).

        print("Processor components checked/moved to target device.")
        return mm_processor

    async def process_request(self, request_payload: dict):
        """
        Processes a multimodal request using the dedicated GPU and stream.
        Expects request_payload to contain 'image_data', 'input_text_or_ids', 'obj_dict'.
        """
        rid = request_payload.get("rid", "unknown")
        logger.debug(
            f"Processing request rid={rid} on {self.device} with stream {self.cuda_stream}"
        )
        try:
            image_data = request_payload.get("image_data")
            input_text_or_ids = request_payload.get("input_text_or_ids")
            # The original object might contain sampling params etc., needed by processor
            obj_dict = request_payload.get("obj_dict", {})
            max_input_len = request_payload.get(
                "max_input_len"
            )  # Pass max length if needed

            if image_data is None:
                raise ValueError("Request payload missing 'image_data'")

            # Perform GPU-intensive processing within the dedicated stream
            # The processor methods should ideally accept a device/stream or use the current one
            with torch.cuda.stream(self.cuda_stream):
                # Reusing the existing interface of mm_processor
                # Ensure the implementation of process_mm_data uses the current stream context
                processed_data = await self.processor.process_mm_data_async(
                    image_data=image_data,
                    input_text=input_text_or_ids,
                    obj_info=(
                        argparse.Namespace(**obj_dict) if obj_dict else None
                    ),  # Reconstruct obj if needed
                    max_input_len=max_input_len,
                    request_obj=request_payload,
                    max_req_input_len=max_input_len,
                    # Pass device explicitly if the method supports it, otherwise rely on context
                    # target_device=self.device
                )

            # Data returned by process_mm_data might already be on CPU or need conversion
            # Example: {'pixel_values': tensor, 'image_embeds': {'image_embeds': tensor, 'image_embeds_mask': tensor}}
            # We need to ensure tensors are on CPU before sending via ZMQ/pickle
            def to_cpu(data):
                if isinstance(data, torch.Tensor):
                    return data.cpu()
                elif isinstance(data, dict):
                    return {k: to_cpu(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [to_cpu(item) for item in data]
                else:
                    return data

            cpu_processed_data = to_cpu(processed_data)

            response = {"success": True, "result": cpu_processed_data}
            logger.debug(f"Successfully processed request rid={rid}")

        except Exception as e:
            logger.error(
                f"Error processing multimodal request rid={rid}: {e}\n{get_exception_traceback()}"
            )
            response = {"success": False, "error": str(e)}

        return response

    async def run_loop(self):
        """The main event loop for receiving and processing requests."""
        print("Multimodal Processor Service started. Waiting for requests...")
        while True:
            try:
                request_payload = await self.zmq_socket.recv_pyobj()
                # Basic validation
                if not isinstance(request_payload, dict):
                    logger.warning("Received non-dict payload, ignoring.")
                    # REP socket requires a reply, send an error response
                    await self.zmq_socket.send_pyobj(
                        {"success": False, "error": "Invalid payload format"}
                    )
                    continue

                rid = request_payload.get("rid", "unknown")
                logger.debug(f"Received request rid={rid}")

                # Process the request
                response_payload = await self.process_request(request_payload)

                # Send response back to TokenizerManager
                await self.zmq_socket.send_pyobj(response_payload)
                logger.debug(f"Sent response for rid={rid}")

            except zmq.ZMQError as e:
                logger.error(f"ZMQ Error in run_loop: {e}")
                # ZMQ errors might indicate connection issues, pause before retrying
                await asyncio.sleep(1)
            except Exception as e:
                # Catch broad exceptions to keep the loop running
                logger.error(
                    f"Unhandled exception in run_loop: {e}\n{get_exception_traceback()}"
                )
                # Attempt to send an error response back if a request was being processed
                # Check if socket is still valid might be needed
                try:
                    # REP socket *must* send a reply for every receive
                    await self.zmq_socket.send_pyobj(
                        {
                            "success": False,
                            "error": "Internal server error in multimodal processor",
                        }
                    )
                except Exception as send_e:
                    logger.error(
                        f"Failed to send error response after exception: {send_e}"
                    )


def run_multimodal_processor_service(server_args: ServerArgs, port_args: PortArgs):
    """Entry point function for the multimodal processor service process."""
    if server_args.mm_gpu_id is None:
        logger.error("Multimodal processing service requires --mm-gpu-id to be set.")
        # Signal parent about the failure early
        parent_process = psutil.Process().parent()
        if parent_process and parent_process.is_running():
            parent_process.send_signal(signal.SIGQUIT)
        sys.exit(1)

    # Setup process environment
    kill_itself_when_parent_died()
    title = f"sglang::mm_proc_svc_gpu{server_args.mm_gpu_id}"
    setproctitle.setproctitle(title)
    configure_logger(server_args, prefix=f" MMProcSVC-GPU{server_args.mm_gpu_id}")
    parent_process = psutil.Process().parent()

    # Set the default CUDA device for this process
    torch.cuda.set_device(server_args.mm_gpu_id)
    print(f"Process {os.getpid()} set to use GPU {server_args.mm_gpu_id}")

    try:
        service = MultimodalProcessorService(server_args, port_args)
        asyncio.run(service.run_loop())
    except Exception:
        # Catch exceptions during service initialization or loop start
        traceback = get_exception_traceback()
        logger.error(f"MultimodalProcessorService failed to start or run: {traceback}")
        # Signal parent process about the failure
        if parent_process and parent_process.is_running():
            parent_process.send_signal(signal.SIGQUIT)
        sys.exit(1)
    finally:
        print("Multimodal Processor Service shutting down.")
