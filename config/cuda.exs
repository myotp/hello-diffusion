import Config

# https://elixirforum.com/t/livebook-cuda-12-2-xla-out-of-memory-at-11006mib/58054/2
# Load the parameters into the CPU
config :nx, :default_backend, {EXLA.Backend, client: :host}
# Enable lazy transfers in serving defn options
config :nx, :default_defn_options, compiler: EXLA

# https://hexdocs.pm/exla/EXLA.html#module-clients
# config :exla, :default_client, :cuda

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: true, memory_fraction: 0.95]
