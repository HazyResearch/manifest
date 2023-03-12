import asyncio
import time

from manifest import Manifest


def main():

    manifest = Manifest(
        client_name="openaichat",
    )

    print("Running in serial")
    prompts = [f"Tell me something interesting about {i}" for i in range(50)]
    st = time.time()
    for pmt in prompts:
        _ = manifest.run(pmt)
    print(f"For loop: {time.time() - st :.2f}")

    print("Running with async")
    st = time.time()
    _ = asyncio.run(manifest.arun_batch(prompts, max_tokens=30))
    print(f"Async loop: {time.time() - st :.2f}")


if __name__ == "__main__":
    main()
