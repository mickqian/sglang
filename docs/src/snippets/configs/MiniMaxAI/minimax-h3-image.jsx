export const config = (() => {
  const uncachedServer = (s) => s.hw === "h200" && Number(s.nodes) === 1
    && Number(s.gpus_per_node) === 1 && s.placement === "auto" && s.encoder === "auto"
    && s.adaln === "auto" && s.warmup === "server";
  const uncachedWorkload = (s) => uncachedServer(s) && s.mode === "edit"
    && s.size === "2048x2048" && s.references === "one" && s.quality === "lossless";
  const guide4kWorkload = (s) => s.hw === "h200" && (
    (s.mode === "t2i" && ["4736x3552", "6272x2688"].includes(s.size))
    || (s.mode === "edit" && s.size === "4096x4096" && s.references === "one")
  );
  const config = {
  modelName: "MiniMax-H3-Image",
  runModes: ["python"],
  supportedHardware: ["b300", "b200", "h200", "h100", "dgx-spark", "rtxpro6000"],
  hardware: [
    { id: "dgx-spark", label: "DGX Spark", vram: "128GB unified", vendor: "blackwell" },
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
      id: "encoder", title: "Encoder parallelism", scope: "serve", default: "auto",
      description: "Auto uses the runtime policy. Fold shards the large text encoder across the GPUs serving a request.",
      options: [
        { id: "auto", label: "Auto" },
        { id: "fold", label: "Fold", flags: ["--encoder-parallel fold"] },
      ],
    },
    {
      id: "adaln", title: "AdaLN projections", scope: "serve", default: "auto",
      description: "Auto precomputes timestep projections on H200 and B300, saving about 24.6–24.7 GiB on one GPU. New plans read the checkpoint; keep weights to avoid that cold-request cost.",
      options: [
        { id: "auto", label: "Auto", recommended: true },
        { id: "online", label: "Precompute", flags: ["--minimax-h3-adaln-online true"] },
        { id: "resident", label: "Keep weights", flags: ["--minimax-h3-adaln-online false"] },
      ],
    },
    {
      id: "warmup", title: "Startup warmup", scope: "serve", default: "server",
      description: "Prepare both task grids at startup. Skipping warmup starts serving sooner but moves initialization into the first request.",
      options: [
        { id: "server", label: "Prepare at startup", recommended: true },
        { id: "off", label: "Skip", flags: ["--warmup-mode off"] },
      ],
    },
    {
      id: "conditioning", title: "Conditioning reuse", scope: "serve", default: "on",
      description: "Reuse exact text conditioning and VAE moments for matching inputs. Disable it to measure uncached requests; sampling still uses the request's generator.",
      options: [
        { id: "on", label: "Exact reuse", recommended: true },
        { id: "off", label: "Disabled", flags: ["--disable-conditioning-cache"], soft: (s) => !uncachedWorkload(s), softReason: "This uncached server recipe has not been validated with the selected workload." },
      ],
    },
    {
      id: "quality", title: "Quality", scope: "request", default: "lossless",
      description: "Lossless enables native fusion and batched VAE tiles. Exact retains the release arithmetic and tile scheduling; both use six steps.",
      options: [{ id: "lossless", label: "Lossless", recommended: true }, { id: "exact", label: "Exact" }],
    },
    {
      id: "size", title: "Output size", scope: "request", default: "2048x2048",
      description: "Smaller outputs still use the default 2048² minimum generation canvas, then resize. Size does not change the six-step schedule.",
      options: [
        { id: "2048x2048", label: "2048 × 2048" },
        { id: "1024x1024", label: "1024 × 1024" },
        ...["4096x4096", "4736x3552", "3552x4736", "5024x3360", "3360x5024", "5472x3072", "3072x5472", "6272x2688"].map((id) => ({
          id, label: id.replace("x", " × "), soft: (s) => !guide4kWorkload({ ...s, size: id }),
          softReason: "This size has not been measured with the selected hardware, task and reference count.",
        })),
      ],
    },
    {
      id: "seed", title: "Seed", scope: "request", kind: "number", default: 42,
      min: 0, max: 4294967295, step: 1, unit: "",
      description: "Use the same seed, inputs, checkpoint and server settings to reproduce a request. The generator stays on the CPU.",
    },
    {
      id: "references", title: "Reference images", scope: "request", default: "one",
      showWhen: (s) => s.mode === "edit",
      description: "Upload references in prompt order: <Picture 1>, then <Picture 2>. Each file is encoded separately.",
      options: [{ id: "one", label: "One image" }, { id: "two", label: "Two images" }],
    },
  ],
  commandBuilder: {
    defaultSelection: { hw: "b300", nodes: 1, gpus_per_node: 1, topology_mode: "auto", tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
    resource: {
      limits: {
        nodes: { min: 1, max: 1 },
        gpus_per_node: { min: 1, max: 8, disabledWhen: (s) => s.hw === "dgx-spark", disabledValue: 1 },
      },
      verifiedRecipes: [
        { id: "spark-single-derived", hw: "dgx-spark", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "auto", encoder: "auto", adaln: "online", default: true, unverified: true },
        { id: "b300-single", hw: "b300", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "auto", encoder: "auto" },
        { id: "b300-single-resident", hw: "b300", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", encoder: "auto" },
        ...[[2, 1, 1], [1, 2, 1], [1, 1, 2]].map(([tp_size, ulysses_degree, ring_degree]) => ({ id: `b300-two-${tp_size}-${ulysses_degree}-${ring_degree}`, hw: "b300", nodes: 1, gpus_per_node: 2, tp_size, ulysses_degree, ring_degree, placement: "resident", encoder: "fold" })),
        ...["auto", "resident", "offload"].map((placement) => ({ id: `h200-single-${placement}`, hw: "h200", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement, encoder: "auto" })),
        ...[[2, 1, 1], [1, 2, 1], [1, 1, 2]].flatMap(([tp_size, ulysses_degree, ring_degree]) => ["auto", "resident"].map((placement) => ({ id: `h200-two-${tp_size}-${ulysses_degree}-${ring_degree}-${placement}`, hw: "h200", nodes: 1, gpus_per_node: 2, tp_size, ulysses_degree, ring_degree, placement, encoder: "fold" }))),
      ].map((recipe) => ({
        adaln: "auto",
        conditioning: "on",
        ...recipe,
        verifiedWhen: (s) => config.commandBuilder.resolveDeployment({
          ...s, ...recipe, topology_mode: "manual", adaln: recipe.adaln || "auto", warmup: "server", conditioning: "on",
        }).verified,
      })),
      recommendedRecipe: (s) => {
        const t = config.commandBuilder.resource.autoTopology(s);
        return config.commandBuilder.resource.verifiedRecipes.find((r) =>
          r.hw === s.hw && r.nodes === Number(s.nodes) && r.gpus_per_node === Number(s.gpus_per_node)
          && r.tp_size === t.tp_size && r.ulysses_degree === t.ulysses_degree && r.ring_degree === t.ring_degree
        );
      },
      autoTopology: (s) => ({ tp_size: 1, ulysses_degree: Number(s.gpus_per_node), ring_degree: 1 }),
      validateTopology: (s, t) => {
        const degrees = [t.tp_size, t.ulysses_degree, t.ring_degree];
        const errors = [];
        if (s.hw === "dgx-spark" && Number(s.gpus_per_node) !== 1) errors.push("This DGX Spark recipe uses one node with one GB10 GPU.");
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
      const recipe = resource.verifiedRecipes.find((r) => r.hw === s.hw && r.gpus_per_node === Number(s.gpus_per_node) && r.tp_size === topology.tp_size && r.ulysses_degree === topology.ulysses_degree && r.ring_degree === topology.ring_degree && r.placement === s.placement && r.encoder === s.encoder);
      const errors = resource.validateTopology(s, topology);
      const measured2k = s.size === "2048x2048" && (Number(s.gpus_per_node) === 1 || s.quality === "lossless");
      const measured4k = guide4kWorkload(s) && s.placement === "auto"
        && s.adaln !== "resident" && s.warmup === "server";
      const serveVerified = !!recipe && !recipe.unverified && !errors.length
        && (s.conditioning !== "off" || uncachedServer(s))
        && (s.warmup !== "off" || (s.hw === "h200" && Number(s.gpus_per_node) === 1));
      const verified = serveVerified && (s.conditioning === "off" ? uncachedWorkload(s) : measured2k || measured4k);
      const flags = ["--model-path {{MODEL_NAME}}", "--backend sglang"];
      if (Number(s.gpus_per_node) > 1) flags.push(`--num-gpus ${s.gpus_per_node}`, `--tp-size ${topology.tp_size}`, `--ulysses-degree ${topology.ulysses_degree}`, `--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw }, nnodes: 1, verified, flags,
        builder: {
          topology, errors,
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          resolvedSettings: {
            placement: s.hw === "dgx-spark" && s.placement === "auto" ? "Component offload" : undefined,
            adaln: s.adaln === "resident" || (s.adaln === "auto" && !["b300", "h200"].includes(s.hw)) ? "Keep weights" : "Precompute",
          },
          warnings: s.hw === "dgx-spark"
            ? ["Derived for one 128 GB Spark. CPU and GPU share memory; component offload does not add capacity. GB10 memory use and latency are unmeasured; start with 2048 × 2048."]
            : verified ? [] : ["This exact hardware, topology, placement and workload recipe is not yet verified."],
          verification: { serve: errors.length ? "error" : serveVerified ? "verified" : "unverified", request: errors.length ? "error" : verified ? "verified" : "unverified" },
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
    IMAGE_PATH_2: { target: "curl", label: "Second reference image", default: "/path/to/reference-2.png", showWhen: (s) => s.mode === "edit" && s.references === "two" },
  },
  curl: (s) => {
    const save = " | python3 -c 'import base64,json,sys; open(\"generated.png\",\"wb\").write(base64.b64decode(json.load(sys.stdin)[\"data\"][0][\"b64_json\"]))'";
    const references = s.references === "two" ? " \\\n  -F 'image=@{{IMAGE_PATH_2}}'" : "";
    const prompt = s.references === "two"
      ? "Keep the scene from <Picture 1> and use the color palette of <Picture 2>."
      : "Keep the scene and change the sky to a warm sunset.";
    if (s.mode === "edit") return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
  -F 'model={{MODEL_NAME}}' \\
  -F 'image=@{{IMAGE_PATH}}'${references} \\
  -F 'prompt=${prompt}' \\
  -F 'size=${s.size}' -F 'quality=${s.quality}' -F 'seed=${s.seed}' -F 'generator_device=cpu' \\
  -F 'response_format=b64_json' -F 'output_format=png'${save}`;
    return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A quiet London street after rain, warm shop lights reflected on the pavement, detailed photograph", size: s.size, quality: s.quality, seed: Number(s.seed), generator_device: "cpu", response_format: "b64_json", output_format: "png" }, null, 2)}'${save}`;
  },
  cells: [],
  };
  return config;
})();
