from manifest import Manifest


manifest = Manifest(
    client_name="huggingface",
    client_connection="http://127.0.0.1:5001",
)

out = manifest.run("Why is the grass green?")

print(out)
