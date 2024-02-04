defmodule HelloDiffusion.Model do
  # https://github.com/elixir-nx/bumblebee/blob/main/notebooks/stable_diffusion.livemd
  def load() do
    repo_id = "CompVis/stable-diffusion-v1-4"
    opts = [params_variant: "fp16", type: :bf16]

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})
    {:ok, clip} = Bumblebee.load_model({:hf, repo_id, subdir: "text_encoder"}, opts)
    {:ok, unet} = Bumblebee.load_model({:hf, repo_id, subdir: "unet"}, opts)

    {:ok, vae} =
      Bumblebee.load_model({:hf, repo_id, subdir: "vae"}, [architecture: :decoder] ++ opts)

    {:ok, scheduler} = Bumblebee.load_scheduler({:hf, repo_id, subdir: "scheduler"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo_id, subdir: "feature_extractor"})
    {:ok, safety_checker} = Bumblebee.load_model({:hf, repo_id, subdir: "safety_checker"}, opts)

    serving =
      Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
        num_steps: 20,
        num_images_per_prompt: 1,
        safety_checker: safety_checker,
        safety_checker_featurizer: featurizer,
        compile: [batch_size: 1, sequence_length: 60],
        # Option 1
        # preallocate_params: true,
        # defn_options: [compiler: EXLA]
        # Option 2 (reduces GPU usage, but runs noticeably slower)
        defn_options: [compiler: EXLA, lazy_transfers: :always]
      )

    {:ok, serving}
  end

  def start_serving_process(serving) do
    Nx.Serving.start_link(name: MyDiffusion, serving: serving)
  end

  def run(prompt, neg_propmt) do
    output = Nx.Serving.batched_run(MyDiffusion, %{prompt: prompt, negative_prompt: neg_propmt})
    output.results
  end
end
