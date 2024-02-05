defmodule HelloDiffusion.Demo do
  def load_model() do
    HelloDiffusion.Model.load()
  end

  def run(serving, prompt, filename) do
    x = Nx.Serving.run(serving, prompt)
    t = hd(x.results).image

    {:ok, vix_image} = Image.from_nx(t)
    {:ok, bin} = Image.write(vix_image, :memory, suffix: ".jpg")
    File.write(filename, bin)
  end
end
