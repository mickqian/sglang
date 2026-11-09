export const config = {
  modelName: "MiniMax-H3-Image",
  supportedHardware: ["b300", "b200", "h200", "h100", "rtxpro6000"],
  hardware: [
    { id: "rtxpro6000", label: "RTX PRO 6000 Blackwell", vram: "96GB", vendor: "consumer" },
  ],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "mode", title: "Task", scope: "base", default: "t2i",
      description: "One checkpoint serves text-to-image and ordered image-reference editing.",
      options: [{ id: "t2i", label: "Text to image" }, { id: "edit", label: "Image editing" }],
    },
    {
      id: "placement", title: "Placement", scope: "serve", default: "auto",
      description: "Auto chooses component placement. Resident needs room for all weights and activations; component offload transfers whole modules.",
      options: [
        { id: "auto", label: "Auto", recommended: true },
        { id: "resident", label: "Resident", flags: ["--component-residency all=resident"] },
        { id: "offload", label: "Component offload", flags: ["--component-residency all=component-offload"] },
      ],
    },
    {
      id: "size", title: "Output size", scope: "request", default: "2048x2048",
      description: "Smaller outputs still use the default 2048² minimum generation canvas, then resize. Size does not change the six-step schedule.",
      options: [
        { id: "2048x2048", label: "2048 × 2048" },
        { id: "1024x1024", label: "1024 × 1024" },
        { id: "4736x3552", label: "4736 × 3552", soft: true, softReason: "This high-resolution workload is not yet measured." },
      ],
    },
  ],
  commandBuilder: {
    defaultSelection: { hw: "b300", nodes: 1, gpus_per_node: 1, topology_mode: "auto", tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
      verifiedRecipes: [],
      autoTopology: (s) => ({ tp_size: 1, ulysses_degree: Number(s.gpus_per_node), ring_degree: 1 }),
      validateTopology: (s, t) => {
        const degrees = [t.tp_size, t.ulysses_degree, t.ring_degree];
        const errors = [];
        if (!degrees.every((n) => Number.isInteger(n) && n > 0)) errors.push("Parallel degrees must be positive integers.");
        if (Number(s.nodes) !== 1 || Number(s.gpus_per_node) !== degrees.reduce((a, b) => a * b, 1)) errors.push("Use one node with GPU count equal to TP × Ulysses × Ring.");
        if (![1, 2, 4, 8].includes(t.tp_size)) errors.push("TP must divide both the DiT and text-encoder head counts: choose 1, 2, 4, or 8.");
        if (56 % (t.tp_size * t.ulysses_degree)) errors.push("TP × Ulysses must divide the 56 DiT attention heads.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : resource.autoTopology(s);
      const flags = ["--model-path {{MODEL_NAME}}", "--warmup-mode off"];
      if (Number(s.gpus_per_node) > 1) flags.push(`--num-gpus ${s.gpus_per_node}`, `--tp-size ${topology.tp_size}`, `--ulysses-degree ${topology.ulysses_degree}`, `--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw }, nnodes: 1, verified: false, flags,
        builder: {
          topology, errors: resource.validateTopology(s, topology),
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          warnings: ["This exact hardware, topology, placement and workload recipe is not yet verified."],
          verification: { serve: "unverified", request: "unverified" },
        },
      };
    },
  },
  modelNames: { default: "MiniMaxAI/MiniMax-H3-Image" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    IMAGE_PATH: { target: "curl", label: "Reference image", default: "/path/to/reference.png", showWhen: (s) => s.mode === "edit" },
  },
  curl: (s) => {
    const save = " | python3 -c 'import base64,json,sys; open(\"generated.png\",\"wb\").write(base64.b64decode(json.load(sys.stdin)[\"data\"][0][\"b64_json\"]))'";
    if (s.mode === "edit") return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
  -F 'model={{MODEL_NAME}}' \\
  -F 'image=@{{IMAGE_PATH}}' \\
  -F 'prompt=Keep the scene and change the sky to a warm sunset.' \\
  -F 'size=${s.size}' -F 'seed=42' -F 'generator_device=cpu' \\
  -F 'response_format=b64_json' -F 'output_format=png'${save}`;
    return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A quiet London street after rain, warm shop lights reflected on the pavement, detailed photograph", size: s.size, seed: 42, generator_device: "cpu", response_format: "b64_json", output_format: "png" }, null, 2)}'${save}`;
  },
  cells: [],
};
